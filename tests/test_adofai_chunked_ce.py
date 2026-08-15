"""Train CE must not materialize full [B, T, V] logits.

Field data (main ``b11f4d4``, PR #13, 4×A100-80GB, batch 8, tgt 8192,
vocab 95471):

- ``mixed_precision=bf16`` and ``strip_fp32_output_conversion`` work.
- First step still OOM at ``modeling_mapperatorinator.py:217``:
  ``loss_fn(swapaxes(output.logits, 1, -1), labels)``
- Tried 11.65 GiB, 3.49 GiB free, 75.75/79.25 GiB used
  (57.17 alloc + 17.46 reserved). 8 × 8192 × 95471 × 2 bytes = 11.65 GiB.

Cause: ``self.transformer.forward`` runs ``proj_out`` / ``lm_head`` on the
full decoder hidden state, then CE builds softmax workspace on that same
``[B, T, V]`` tensor. Returning loss-only (PR #13) does not help: the
tensor already exists before the return.

Train must apply ``lm_head`` in chunks over T (or fused CE) so loss/backward
never hold a full ``[B, T, V]`` tensor. Eval/generate may still use logits.
These tests must fail if train CE is still the full ``swapaxes(logits)`` path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELING = (
    REPO_ROOT / "osuT5" / "osuT5" / "model" / "modeling_mapperatorinator.py"
)
CHUNKED = REPO_ROOT / "osuT5" / "osuT5" / "model" / "chunked_cross_entropy.py"


def _load_chunked_ce():
    import importlib.util

    spec = importlib.util.spec_from_file_location("osuT5_chunked_cross_entropy", CHUNKED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forward_src() -> str:
    src = MODELING.read_text(encoding="utf-8")
    return src.split("def forward(")[1].split("def prepare_inputs")[0]


def _compact(text: str) -> str:
    return "".join(text.split())


def _code_only(text: str) -> str:
    """Drop comments so the OOM-path check does not match docstrings."""
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0]
        lines.append(line)
    return "\n".join(lines)


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


def test_train_ce_is_not_full_swapaxes_logits_path():
    """Fails if train CE is still loss_fn(swapaxes(output.logits), labels)."""
    src = MODELING.read_text(encoding="utf-8")
    forward_src = _code_only(_forward_src())
    before_train_return = forward_src.split("if self.training")[0]
    compact_before = _compact(before_train_return)

    assert "chunked_linear_cross_entropy" in src
    assert "decoder_hidden_states" in src
    assert "self.transformer.forward" not in compact_before
    assert "swapaxes(output.logits" not in compact_before
    assert "self.loss_fn(torch.swapaxes" not in compact_before


def test_training_return_still_omits_logits():
    """PR #13: convert_to_fp32 must not see a logits field on the train path."""
    src = MODELING.read_text(encoding="utf-8")
    assert "if self.training" in src
    assert "return Seq2SeqLMOutput(loss=loss)" in src
    train_return = (
        src.split("if self.training")[1].split("return", 1)[1].split("\n", 1)[0]
    )
    assert "Seq2SeqLMOutput(loss=loss)" in train_return
    assert "logits" not in train_return


def test_pr13_strip_and_pr12_bf16_and_pr11_nccl_stay():
    src = (REPO_ROOT / "osuT5" / "train.py").read_text(encoding="utf-8")
    assert "strip_fp32_output_conversion" in src
    assert src.index("accelerator.prepare(") < src.index(
        "strip_fp32_output_conversion(model)"
    )
    assert "resolve_mixed_precision" in src
    assert "configure_nccl_for_pcie_multigpu" in src


def test_chunked_ce_linear_never_materializes_full_btv(monkeypatch):
    """Train CE path must never allocate/return a full [B, T, V] logit tensor."""
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    helper = _load_chunked_ce()

    batch, tgt, dim, vocab = 2, 32, 8, 48
    chunk_size = 8
    hidden = torch.randn(batch, tgt, dim, dtype=torch.bfloat16, requires_grad=True)
    lm_head = torch.nn.Linear(dim, vocab, bias=False)
    lm_head.weight.data = lm_head.weight.data.to(torch.bfloat16)
    lm_head.weight.requires_grad_(True)
    labels = torch.randint(0, vocab, (batch, tgt))
    labels[0, 0] = helper.LABEL_IGNORE_ID

    linear_shapes: list[tuple[int, ...]] = []
    ce_shapes: list[tuple[int, ...]] = []
    real_linear = torch.nn.functional.linear
    real_ce = torch.nn.functional.cross_entropy

    def _spy_linear(input, weight, bias=None):
        out = real_linear(input, weight, bias)
        linear_shapes.append(tuple(out.shape))
        return out

    def _spy_ce(input, target, *args, **kwargs):
        ce_shapes.append(tuple(input.shape))
        return real_ce(input, target, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "linear", _spy_linear)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", _spy_ce)

    loss = helper.chunked_linear_cross_entropy(
        hidden, lm_head, labels, chunk_size=chunk_size
    )
    assert loss.ndim == 0
    assert not any(shape == (batch, tgt, vocab) for shape in linear_shapes), (
        f"lm_head materialized full [B, T, V] logits: {linear_shapes}"
    )
    assert all(shape[1] <= chunk_size for shape in linear_shapes), (
        f"lm_head chunk exceeded chunk_size={chunk_size}: {linear_shapes}"
    )
    assert not any(shape == (batch, vocab, tgt) for shape in ce_shapes), (
        f"CE still saw full swapaxes(logits) [B, V, T]: {ce_shapes}"
    )
    assert not any(
        shape == (batch * tgt, vocab) or shape == (vocab, batch * tgt)
        for shape in ce_shapes
    ), f"CE still saw a full flattened [B*T, V] tensor: {ce_shapes}"
    assert ce_shapes, "cross_entropy was never called"
    loss.backward()
    assert hidden.grad is not None
    assert lm_head.weight.grad is not None


def test_chunked_ce_matches_full_ce_on_small_tensors():
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    helper = _load_chunked_ce()

    torch.manual_seed(0)
    batch, tgt, dim, vocab = 3, 20, 6, 11
    hidden = torch.randn(batch, tgt, dim, requires_grad=True)
    lm_head = torch.nn.Linear(dim, vocab, bias=False)
    labels = torch.randint(0, vocab, (batch, tgt))
    labels[:, -1] = helper.LABEL_IGNORE_ID
    class_weight = torch.ones(vocab)
    class_weight[0] = 2.5
    sample_weights = torch.tensor([1.0, 0.5, 1.25])

    logits = torch.nn.functional.linear(hidden, lm_head.weight)
    ref = torch.nn.functional.cross_entropy(
        logits.swapaxes(1, -1),
        labels,
        weight=class_weight,
        ignore_index=helper.LABEL_IGNORE_ID,
        reduction="none",
    )
    ref = (ref * sample_weights.unsqueeze(1)).sum() / (labels != helper.LABEL_IGNORE_ID).sum()

    got = helper.chunked_linear_cross_entropy(
        hidden,
        lm_head,
        labels,
        class_weight=class_weight,
        sample_weights=sample_weights,
        chunk_size=7,
    )
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_decoder_hidden_states_skips_proj_out():
    """Train must run transformer.model, not transformer.forward / proj_out."""
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    helper = _load_chunked_ce()

    hidden = torch.randn(2, 5, 4)

    class _Body(torch.nn.Module):
        def forward(self, **kwargs):
            return (kwargs["decoder_inputs_embeds"],)

    class _Head(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("proj_out/lm_head must not run for train hidden states")

    class _Full(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.proj_out(self.model(**kwargs)[0]))

    transformer = _Full()
    transformer.model = _Body()
    transformer.proj_out = _Head()

    out = helper.decoder_hidden_states(
        transformer, {"decoder_inputs_embeds": hidden, "use_cache": True}
    )
    assert out.shape == hidden.shape
    assert torch.equal(out, hidden)


def test_train_path_helper_does_not_return_logits():
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    helper = _load_chunked_ce()

    hidden = torch.randn(2, 6, 4, requires_grad=True)
    lm_head = torch.nn.Linear(4, 9, bias=False)
    labels = torch.randint(0, 9, (2, 6))
    loss = helper.chunked_linear_cross_entropy(hidden, lm_head, labels, chunk_size=3)
    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert getattr(loss, "logits", None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
