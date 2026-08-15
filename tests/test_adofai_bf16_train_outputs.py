"""8192-tgt train outputs must stay bf16 through the train step.

Field data (main ``ce9337a``, PR #12, 4×A100-80GB, batch 8, compile=false,
tgt 8192):

- ``mixed_precision=bf16`` now reaches Accelerator (PR #12).
- DDP 427-vs-427. First forward still OOM: tried 17.48 GiB, 9.21 GiB free,
  70.03/79.25 GiB used. Same peak as ``mixed_precision=no``.
- Stack: accelerate ``convert_to_fp32`` / ``tensor.float()`` after
  ``loss, stats = forward(...)`` in ``osuT5/osuT5/utils/train_utils.py``.

Cause: ``Accelerator.prepare`` wraps ``model.forward`` with
``convert_outputs_to_fp32`` (accelerate 1.12.0
``utils/operations.py`` ``ConvertOutputsToFp32.__call__`` →
``convert_to_fp32`` → ``tensor.float()``). ``Seq2SeqLMOutput`` is a
Mapping, so every bf16 field is cloned to fp32 — including
``logits`` ``[B, 8192, ~95471]`` returned by
``modeling_mapperatorinator.py``. The config flag is not enough.

These tests must fail if that clone comes back.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bf16_train_outputs():
    import importlib.util

    path = REPO_ROOT / "osuT5" / "osuT5" / "utils" / "bf16_train_outputs.py"
    spec = importlib.util.spec_from_file_location("osuT5_bf16_train_outputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset_accelerate_state():
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


class _MappingOutput(dict):
    """Mapping like Seq2SeqLMOutput so convert_to_fp32 walks ``logits``."""

    __getattr__ = dict.__getitem__


class _TgtLogitsStub:
    """Stand-in for Mapperatorinator train forward: bf16 [B, T, V] logits."""

    def __init__(self, torch, *, return_logits=True):
        from torch import nn

        class _Mod(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
                self.return_logits = return_logits

            def forward(self, **batch):
                logits = self.w * torch.ones(
                    2, 16, 32, dtype=torch.bfloat16, device=self.w.device
                )
                loss = logits.mean()
                if self.return_logits:
                    return _MappingOutput(loss=loss, logits=logits)
                return _MappingOutput(loss=loss)

        self.module = _Mod()


def _train_forward(model, batch):
    """Same shape as ``train_utils.forward``: ``loss, stats = forward(...)``."""
    outputs = model(**batch)
    loss = outputs.loss
    return loss, {"loss": loss.detach()}


def test_adofai_v31_width_and_train_command_unchanged():
    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "train" / "adofai_v31.yaml").read_text(encoding="utf-8")
    )
    assert data["mixed_precision"] == "bf16"
    assert data["precision"] == "bf16"
    assert data["model"]["cond_size"] == 128
    assert data["data"]["tgt_seq_len"] == 8192
    assert data["optim"]["batch_size"] == 32
    assert "python osuT5/train.py -cn adofai_v31" in (
        REPO_ROOT / "adofai" / "train.py"
    ).read_text(encoding="utf-8")


def test_train_py_strips_fp32_conversion_after_prepare():
    """prepare() installs convert_outputs_to_fp32; strip must run after that."""
    src = (REPO_ROOT / "osuT5" / "train.py").read_text(encoding="utf-8")
    assert "strip_fp32_output_conversion" in src
    # Call after prepare(), not the import line.
    assert src.index("accelerator.prepare(") < src.index(
        "strip_fp32_output_conversion(model)"
    )
    assert "resolve_mixed_precision" in src
    assert "configure_nccl_for_pcie_multigpu" in src


def test_mapperatorinator_training_forward_omits_logits():
    """Returning logits=output.logits is what convert_to_fp32 clones to fp32."""
    src = (
        REPO_ROOT / "osuT5" / "osuT5" / "model" / "modeling_mapperatorinator.py"
    ).read_text(encoding="utf-8")
    assert "if self.training" in src
    after_loss = src.split("if labels is not None:")[1]
    train_branch = after_loss.split("if self.training")[1]
    first_return = train_branch.split("return", 1)[1].split("\n", 1)[0]
    assert "Seq2SeqLMOutput(loss=loss)" in first_return
    assert "logits" not in first_return


def test_train_utils_forward_does_not_float_outputs():
    src = (REPO_ROOT / "osuT5" / "osuT5" / "utils" / "train_utils.py").read_text(
        encoding="utf-8"
    )
    block = src.split("def forward(model")[1].split("def forward_eval")[0]
    assert "model(**batch)" in block
    assert ".float()" not in block
    assert "convert_to_fp32" not in block
    assert "gather" not in block


def test_prepare_without_strip_upcasts_mapping_logits(monkeypatch):
    """Documents accelerate 1.12.0: prepare() + Mapping logits → fp32 clone."""
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    torch = pytest.importorskip("torch")

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_state()
    try:
        accelerator = accelerate.Accelerator(cpu=True, mixed_precision="bf16")
        model = accelerator.prepare(_TgtLogitsStub(torch).module)
        out = model()
        assert out.logits.dtype == torch.float32
        assert out.loss.dtype == torch.float32
    finally:
        _reset_accelerate_state()


def test_train_step_tgt_outputs_stay_bf16(monkeypatch):
    """Train-step path after strip: loss/logits used for backward stay bf16."""
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    torch = pytest.importorskip("torch")
    helper = _load_bf16_train_outputs()

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_state()
    try:
        accelerator = accelerate.Accelerator(cpu=True, mixed_precision="bf16")
        model = accelerator.prepare(_TgtLogitsStub(torch).module)
        helper.strip_fp32_output_conversion(model)

        loss, stats = _train_forward(model, {})
        assert loss.dtype == torch.bfloat16, (
            f"train-step loss was silently upcast to {loss.dtype}"
        )
        assert stats["loss"].dtype == torch.bfloat16

        out = model()
        assert out.logits.dtype == torch.bfloat16, (
            f"8192-tgt stand-in logits were upcast to {out.logits.dtype}"
        )
        loss.backward()
        assert model.w.grad is not None
        assert model.w.grad.dtype == torch.bfloat16
    finally:
        _reset_accelerate_state()


def test_train_step_fails_if_tgt_outputs_silently_upcast(monkeypatch):
    """Spy: tensor.float() on tgt-shaped bf16 outputs is the 80GB OOM."""
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    torch = pytest.importorskip("torch")
    helper = _load_bf16_train_outputs()

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_state()
    try:
        accelerator = accelerate.Accelerator(cpu=True, mixed_precision="bf16")
        model = accelerator.prepare(_TgtLogitsStub(torch).module)
        helper.strip_fp32_output_conversion(model)

        floated_tgt = []
        real_float = torch.Tensor.float

        def _spy_float(self, *args, **kwargs):
            if self.dtype in (torch.bfloat16, torch.float16) and self.numel() >= 2 * 16 * 32:
                floated_tgt.append(tuple(self.shape))
            return real_float(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "float", _spy_float)

        loss, _stats = _train_forward(model, {})
        assert floated_tgt == [], (
            "convert_to_fp32 / tensor.float() cloned tgt forward outputs to "
            f"fp32: {floated_tgt}"
        )
        assert loss.dtype == torch.bfloat16
        loss.backward()
    finally:
        _reset_accelerate_state()


def test_training_seq2seq_output_has_no_logits_tensor():
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    helper = _load_bf16_train_outputs()

    loss = torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)
    out = helper.training_seq2seq_output(loss)
    assert out.loss is loss
    assert out.loss.dtype == torch.bfloat16
    assert out.logits is None
    if isinstance(out, Mapping):
        assert out.get("logits") in (None, )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
