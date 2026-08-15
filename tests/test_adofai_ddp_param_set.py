"""DDP reducer param-set must match across independent v31 constructions.

v31 on 4×A40 died in ``accelerator.prepare()`` → DDP:

    Rank 3 has 427 params, while rank 0 has inconsistent 0 params

PyTorch 2.8 ``_build_params_for_reducer`` counts only ``requires_grad``
tensors. ``print_model_parameters`` sums *all* ``numel()``, so rank 0 can
print Embedding 95471×768 / 478.8M and still hand DDP an empty list.

Two-process Gloo ``accelerator.prepare`` of the real 478.8M model is not
run here: it needs ~4 GiB plus a broadcast of every tensor and is not a
unit-test cost. The cheap Gloo case below uses a tiny stand-in of the
0-vs-N reducer bug; the real v31 model is checked by two sequential
constructions (same signature) plus ``assert_model_ready_for_ddp``.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clear_hydra():
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _compose_adofai_v31():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    _clear_hydra()
    cfg_dir = str(REPO_ROOT / "configs" / "train")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        return OmegaConf.to_object(compose(config_name="adofai_v31"))


def test_train_py_builds_cpu_model_and_asserts_before_prepare():
    """Guard the train.py order that left rank 0 with 0 reducer params."""
    src = (REPO_ROOT / "osuT5" / "train.py").read_text(encoding="utf-8")
    assert "assert_model_ready_for_ddp" in src
    assert 'device="cpu"' in src
    assert "accelerator.wait_for_everyone()" in src
    assert src.index("get_shared_training_state()") < src.index("Accelerator(")
    assert src.index("assert_model_ready_for_ddp") < src.index("accelerator.prepare")
    assert src.index("accelerator.prepare") < src.index("init_trackers")
    assert "python osuT5/train.py -cn adofai_v31" in (
        REPO_ROOT / "adofai" / "train.py"
    ).read_text(encoding="utf-8")


def test_frozen_rank0_is_the_zero_param_ddp_bug_shape():
    torch = pytest.importorskip("torch")
    from osuT5.osuT5.utils.model_utils import (
        assert_identical_ddp_param_sets,
        assert_model_ready_for_ddp,
        ddp_param_signature,
        ddp_reducer_named_parameters,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.lin = torch.nn.Linear(8, 8)

    live = Tiny()
    frozen = Tiny()
    frozen.requires_grad_(False)

    live_sig = assert_model_ready_for_ddp(live)
    assert len(live_sig) > 0
    assert len(ddp_reducer_named_parameters(frozen)) == 0
    with pytest.raises(RuntimeError, match="0 reducer params"):
        assert_model_ready_for_ddp(frozen)

    other_live = Tiny()
    assert_identical_ddp_param_sets(live_sig, ddp_param_signature(other_live))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_ddp_worker(rank: int, world_size: int, port: int, freeze_rank0: bool, result):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        model = torch.nn.Sequential(
            torch.nn.Embedding(16, 8),
            torch.nn.Linear(8, 8),
        )
        if freeze_rank0 and rank == 0:
            model.requires_grad_(False)
        try:
            DistributedDataParallel(model)
            result[rank] = ("ok", sum(1 for p in model.parameters() if p.requires_grad))
        except Exception as exc:  # noqa: BLE001 — worker must report any DDP failure
            result[rank] = ("err", str(exc))
    finally:
        dist.destroy_process_group()


def test_gloo_ddp_two_process_rejects_frozen_rank0():
    """Cheap CPU stand-in for accelerator.prepare on 2 ranks."""
    torch = pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_ddp_worker,
        args=(2, port, True, result),
        nprocs=2,
        join=True,
    )
    messages = " ".join(str(item) for item in result.values())
    assert "0 params" in messages or "inconsistent" in messages or "err" in {
        result[0][0],
        result[1][0],
    }


def test_gloo_ddp_two_process_accepts_identical_unfrozen_models():
    torch = pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_ddp_worker,
        args=(2, port, False, result),
        nprocs=2,
        join=True,
    )
    assert result[0][0] == "ok"
    assert result[1][0] == "ok"
    assert result[0][1] == result[1][1]
    assert result[0][1] > 0


def _build_v31_model():
    import torch
    from osuT5.osuT5.tokenizer import Tokenizer
    from osuT5.osuT5.utils.model_utils import _get_model

    args = _compose_adofai_v31()
    tokenizer = Tokenizer(args)
    model = _get_model(args, tokenizer, dtype=torch.float32, attn_implementation="eager")
    return model, tokenizer


def test_adofai_v31_two_constructions_same_ddp_param_set():
    """Simulates two ranks each calling load_model / _get_model independently."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from osuT5.osuT5.utils.model_utils import (
        assert_identical_ddp_param_sets,
        assert_model_ready_for_ddp,
        ddp_param_signature,
    )

    try:
        model_a, tokenizer_a = _build_v31_model()
    except Exception as exc:  # HF hub / missing backbone config
        pytest.skip(f"cannot construct adofai_v31 model: {exc}")

    sig_a = assert_model_ready_for_ddp(model_a)
    # PR #9 buckets: ~95471, not the 270M raw-range explosion.
    assert 95_000 <= tokenizer_a.vocab_size_out <= 96_000
    embed = dict(sig_a).get("decoder_embedder.weight")
    if embed is None:
        embed = next(shape for name, shape in sig_a if "decoder_embedder" in name)
    assert embed[0] == tokenizer_a.vocab_size_out
    del model_a

    model_b, tokenizer_b = _build_v31_model()
    sig_b = assert_model_ready_for_ddp(model_b)
    assert tokenizer_b.vocab_size_out == tokenizer_a.vocab_size_out
    assert_identical_ddp_param_sets(sig_a, sig_b)
    assert len(sig_b) > 0
    del model_b


def test_v31_yaml_still_cond_size_128_and_official_train_command():
    import yaml

    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "train" / "adofai_v31.yaml").read_text(encoding="utf-8")
    )
    assert data["model"]["cond_size"] == 128
    assert data["data"]["tgt_seq_len"] == 8192
    assert data["optim"]["batch_size"] == 32


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
