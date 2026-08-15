"""DDP param-set + NCCL verify hang on 4×A40.

Field data (main ``a38db9f``, ``num_workers=0``, same 4×A40):

- All 4 ranks printed ``Model loaded`` / ``Embedding(95471, 768)`` first.
- Hung at ``accelerator.prepare`` → DDP ``_verify_param_shape_across_processes``.
- First collective: ``WorkNCCL SeqNum=1 ALLGATHER NumelIn=1 NumelOut=4``,
  timeout 600000 ms. All 4 ranks timed out together (~600065–600086 ms).
- ``427-vs-0`` printed *after* the timeout — garbage from the failed
  ALLGATHER, not a real empty rank-0 model.
- 1419 MiB / 100% util on all 4. Topology PIX/PXB, no NVLink.

427 is the Muon 194 + AdamW 233 optimizer-split of the 478,783,248-param
model. Every rank must still build that same set. The live hang is NCCL
P2P on PIX/PXB during the 1-int verify allgather.

Two-process Gloo wrap of the real 478.8M model is not run here (~4 GiB
plus a full-state broadcast). v31 is checked by two sequential
constructions (427 tensors, ``Embedding(95471, 768)``).
"""

from __future__ import annotations

import os
import socket
from collections import OrderedDict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ddp_utils():
    """Load ddp_utils without importing osuT5.utils (that pulls slider)."""
    import importlib.util

    path = REPO_ROOT / "osuT5" / "osuT5" / "utils" / "ddp_utils.py"
    spec = importlib.util.spec_from_file_location("osuT5_ddp_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_train_py_configures_nccl_before_accelerator_and_gathers_on_gloo():
    """NCCL env + Manager fork + Gloo count gather must happen before DDP wrap."""
    src = (REPO_ROOT / "osuT5" / "train.py").read_text(encoding="utf-8")
    assert "configure_nccl_for_pcie_multigpu" in src
    assert "bind_cuda_device_from_local_rank" in src
    assert "allgather_ddp_reducer_counts" in src
    assert "assert_same_ddp_reducer_counts" in src
    assert "assert_model_ready_for_ddp" in src
    assert "sync_registered_modules" in src
    assert 'device="cpu"' in src
    assert src.index("configure_nccl_for_pcie_multigpu()") < src.index("get_shared_training_state()")
    assert src.index("get_shared_training_state()") < src.index("bind_cuda_device_from_local_rank()")
    assert src.index("bind_cuda_device_from_local_rank()") < src.index("accelerator = Accelerator(")
    assert src.index("model.to(accelerator.device)") < src.index("optimizer = get_optimizer")
    assert src.index("allgather_ddp_reducer_counts") < src.index("accelerator.prepare")
    assert src.index("accelerator.prepare") < src.index("init_trackers")
    assert "python osuT5/train.py -cn adofai_v31" in (
        REPO_ROOT / "adofai" / "train.py"
    ).read_text(encoding="utf-8")


def test_configure_nccl_sets_p2p_level_nvl_when_unset(monkeypatch):
    ddp_utils = _load_ddp_utils()
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_IB_DISABLE", raising=False)
    applied = ddp_utils.configure_nccl_for_pcie_multigpu()
    assert applied["NCCL_P2P_LEVEL"] == "NVL"
    assert applied["NCCL_IB_DISABLE"] == "1"
    assert os.environ["NCCL_P2P_LEVEL"] == "NVL"
    assert os.environ["NCCL_IB_DISABLE"] == "1"


def test_configure_nccl_does_not_override_user_p2p(monkeypatch):
    ddp_utils = _load_ddp_utils()
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    monkeypatch.delenv("NCCL_P2P_LEVEL", raising=False)
    applied = ddp_utils.configure_nccl_for_pcie_multigpu()
    assert "NCCL_P2P_LEVEL" not in applied
    assert "NCCL_IB_DISABLE" not in applied
    assert os.environ["NCCL_P2P_DISABLE"] == "0"
    assert os.environ["NCCL_IB_DISABLE"] == "0"


def test_same_reducer_counts_rejects_zero_and_mismatch():
    ddp_utils = _load_ddp_utils()
    ddp_utils.assert_same_ddp_reducer_counts([427, 427, 427, 427])
    with pytest.raises(RuntimeError, match="0 DDP reducer"):
        ddp_utils.assert_same_ddp_reducer_counts([427, 0, 427, 427])
    with pytest.raises(RuntimeError, match="disagree"):
        ddp_utils.assert_same_ddp_reducer_counts([427, 426, 427, 427])


def test_mapperatorinator_does_not_slot_child_modules():
    """Incomplete __slots__ on child names hides them from DDP's _modules walk."""
    src = (
        REPO_ROOT / "osuT5" / "osuT5" / "model" / "modeling_mapperatorinator.py"
    ).read_text(encoding="utf-8")
    assert "class Mapperatorinator" in src
    assert '__slots__ = ["spectrogram"' not in src
    assert "__slots__ = ['spectrogram'" not in src


def test_empty_modules_is_the_zero_param_ddp_bug_shape():
    """Hardening: empty ``_modules`` is a real 0-param set, not the NCCL hang."""
    torch = pytest.importorskip("torch")
    ddp_utils = _load_ddp_utils()
    assert_model_ready_for_ddp = ddp_utils.assert_model_ready_for_ddp
    sync_registered_modules = ddp_utils.sync_registered_modules

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.lin = torch.nn.Linear(8, 8)

    live = Tiny()
    live_sig = assert_model_ready_for_ddp(live)
    assert len(live_sig) > 0

    emptied = Tiny()
    saved = OrderedDict(emptied._modules)
    emptied._modules = OrderedDict()
    for name, child in saved.items():
        emptied.__dict__[name] = child

    assert list(emptied._modules) == []
    assert sum(1 for _ in emptied.parameters()) == 0
    with pytest.raises(RuntimeError, match="module is empty"):
        assert_model_ready_for_ddp(emptied)

    restored = sync_registered_modules(emptied)
    assert "emb" in restored and "lin" in restored
    assert_model_ready_for_ddp(emptied)
    assert len(ddp_utils.ddp_reducer_named_parameters(emptied)) == len(live_sig)


def test_slotted_child_desync_is_repaired_for_ddp():
    """Slot holds the child, ``_modules`` does not — DDP would see 0 params."""
    torch = pytest.importorskip("torch")
    ddp_utils = _load_ddp_utils()
    assert_model_ready_for_ddp = ddp_utils.assert_model_ready_for_ddp
    sync_registered_modules = ddp_utils.sync_registered_modules

    class Slotted(torch.nn.Module):
        __slots__ = ("lin",)

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)
            object.__setattr__(self, "lin", self._modules["lin"])

    slotted = Slotted()
    child = slotted.lin
    slotted._modules = OrderedDict()
    assert child is object.__getattribute__(slotted, "lin")
    assert sum(1 for _ in slotted.parameters()) == 0
    with pytest.raises(RuntimeError, match="module is empty"):
        assert_model_ready_for_ddp(slotted)

    assert sync_registered_modules(slotted) == ["lin"]
    assert slotted._modules["lin"] is child
    assert_model_ready_for_ddp(slotted)


def test_frozen_rank0_is_not_the_v31_empty_module_bug():
    """Frozen tensors are a different 0-reducer shape; unfreeze cannot fill _modules."""
    torch = pytest.importorskip("torch")
    ddp_utils = _load_ddp_utils()
    assert_identical_ddp_param_sets = ddp_utils.assert_identical_ddp_param_sets
    assert_model_ready_for_ddp = ddp_utils.assert_model_ready_for_ddp
    ddp_param_signature = ddp_utils.ddp_param_signature
    ddp_reducer_named_parameters = ddp_utils.ddp_reducer_named_parameters

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
    assert sum(1 for _ in frozen.parameters()) > 0
    with pytest.raises(RuntimeError, match="0 reducer params"):
        assert_model_ready_for_ddp(frozen)

    other_live = Tiny()
    assert_identical_ddp_param_sets(live_sig, ddp_param_signature(other_live))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_ddp_worker(rank: int, world_size: int, port: int, mode: str, result):
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
        if mode == "empty_rank0" and rank == 0:
            model._modules = OrderedDict()
        elif mode == "freeze_rank0" and rank == 0:
            model.requires_grad_(False)
        try:
            DistributedDataParallel(model)
            result[rank] = ("ok", sum(1 for p in model.parameters() if p.requires_grad))
        except Exception as exc:  # noqa: BLE001 — worker must report any DDP failure
            result[rank] = ("err", str(exc))
    finally:
        dist.destroy_process_group()


def test_gloo_ddp_two_process_rejects_empty_rank0_module():
    """Cheap CPU stand-in: rank 0 ``_modules`` cleared, others keep the full split."""
    torch = pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_ddp_worker,
        args=(2, port, "empty_rank0", result),
        nprocs=2,
        join=True,
    )
    messages = " ".join(str(item) for item in result.values())
    assert "0 params" in messages or "inconsistent" in messages or "empty" in messages.lower() or "err" in {
        result[0][0],
        result[1][0],
    }


def test_gloo_ddp_two_process_rejects_frozen_rank0():
    """Cheap CPU stand-in for accelerator.prepare on 2 ranks."""
    torch = pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_ddp_worker,
        args=(2, port, "freeze_rank0", result),
        nprocs=2,
        join=True,
    )
    messages = " ".join(str(item) for item in result.values())
    assert "0 params" in messages or "inconsistent" in messages or "err" in {
        result[0][0],
        result[1][0],
    }


def _gloo_count_worker(rank: int, world_size: int, port: int, rank0_count: int, result):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ddp_utils = _load_ddp_utils()
        count = rank0_count if rank == 0 else 427
        try:
            gathered = ddp_utils.allgather_ddp_reducer_counts(count)
            ddp_utils.assert_same_ddp_reducer_counts(gathered)
            result[rank] = ("ok", gathered)
        except Exception as exc:  # noqa: BLE001
            result[rank] = ("err", str(exc))
    finally:
        dist.destroy_process_group()


def test_gloo_allgather_reducer_counts_agree():
    """CPU Gloo gather of the 427-split — the check that must not use NCCL."""
    pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_count_worker,
        args=(2, port, 427, result),
        nprocs=2,
        join=True,
    )
    assert result[0][0] == "ok"
    assert result[1][0] == "ok"
    assert result[0][1] == [427, 427]


def test_gloo_allgather_reducer_counts_rejects_mismatch():
    pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_count_worker,
        args=(2, port, 0, result),
        nprocs=2,
        join=True,
    )
    messages = " ".join(str(item) for item in result.values())
    assert "0 DDP reducer" in messages or result[0][0] == "err" or result[1][0] == "err"


def test_gloo_ddp_two_process_accepts_identical_unfrozen_models():
    torch = pytest.importorskip("torch")
    mp = pytest.importorskip("torch.multiprocessing")

    port = _free_port()
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(
        _gloo_ddp_worker,
        args=(2, port, "ok", result),
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
    ddp_utils = _load_ddp_utils()
    assert_identical_ddp_param_sets = ddp_utils.assert_identical_ddp_param_sets
    assert_model_ready_for_ddp = ddp_utils.assert_model_ready_for_ddp
    sync_registered_modules = ddp_utils.sync_registered_modules

    try:
        model_a, tokenizer_a = _build_v31_model()
    except Exception as exc:  # HF hub / missing backbone config
        pytest.skip(f"cannot construct adofai_v31 model: {exc}")

    assert not hasattr(type(model_a), "__slots__") or "transformer" not in getattr(
        type(model_a), "__slots__", ()
    )
    assert sync_registered_modules(model_a) == []
    assert "transformer" in model_a._modules
    sig_a = assert_model_ready_for_ddp(model_a)
    # PR #9 buckets: 95471 out (Whisper embed/lm_head). decoder_embedder is
    # vocab_size_in (out + prefix ranges) and is larger.
    assert tokenizer_a.vocab_size_out == 95471
    assert 95_000 <= tokenizer_a.vocab_size_out <= 96_000
    out_tables = [shape for _, shape in sig_a if shape[:1] == (tokenizer_a.vocab_size_out,)]
    assert (tokenizer_a.vocab_size_out, 768) in out_tables
    # 427 = Muon 194 + AdamW 233 optimizer-split of the 478.8M model.
    assert len(sig_a) == 427
    del model_a

    model_b, tokenizer_b = _build_v31_model()
    sync_registered_modules(model_b)
    sig_b = assert_model_ready_for_ddp(model_b)
    assert tokenizer_b.vocab_size_out == tokenizer_a.vocab_size_out
    assert_identical_ddp_param_sets(sig_a, sig_b)
    assert len(sig_b) == 427
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
