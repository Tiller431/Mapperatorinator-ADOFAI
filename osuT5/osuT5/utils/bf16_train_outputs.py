"""Keep 8192-tgt train outputs in bf16 through accelerate ``prepare()``.

``Accelerator.prepare`` (accelerate 1.12.0 ``accelerator.py``) wraps
``model.forward`` with autocast, then ``convert_outputs_to_fp32``.
``ConvertOutputsToFp32.__call__`` (``utils/operations.py``) runs
``convert_to_fp32`` → ``tensor.float()`` on every bf16 field of a Mapping
output. ``Seq2SeqLMOutput`` is a Mapping, so ``logits`` ``[B, tgt, vocab]``
is cloned to fp32 after ``loss, stats = forward(...)``.

That clone is the 4×A100-80GB first-forward OOM (tried 17.48 GiB) with the
same peak as ``mixed_precision=no``. Config ``mixed_precision: bf16`` is
not enough: the wrapper still upcasts the returned tgt logits.

Training only needs ``loss`` for backward. Do not return the full logits
tensor, and pop the convert wrapper so loss/backward stay bf16.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any


def training_seq2seq_output(loss: Any) -> Any:
    """Loss-only seq2seq output. No logits for ``convert_to_fp32`` to clone."""
    try:
        from transformers.modeling_outputs import Seq2SeqLMOutput
    except ImportError:
        return SimpleNamespace(loss=loss, logits=None)
    return Seq2SeqLMOutput(loss=loss)


def _inner_module(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def strip_fp32_output_conversion(model: Any) -> Any:
    """Remove accelerate ``convert_outputs_to_fp32``; keep the autocast wrap.

    After ``prepare()``, ``model.forward.__func__.__wrapped__`` is
    ``ConvertOutputsToFp32``. Its ``model_forward`` is the autocast-wrapped
    unbound ``forward``. Rebind that and drop ``tensor.float()``.
    """
    inner = _inner_module(model)
    fwd = inner.forward
    func = getattr(fwd, "__func__", fwd)
    wrapped = getattr(func, "__wrapped__", None)
    if wrapped is None:
        return model
    autocast_fwd = getattr(wrapped, "model_forward", None)
    if autocast_fwd is None:
        return model
    if type(wrapped).__name__ != "ConvertOutputsToFp32":
        return model
    if getattr(fwd, "__self__", None) is not None:
        if hasattr(autocast_fwd, "__self__"):
            inner.forward = autocast_fwd
        else:
            inner.forward = MethodType(autocast_fwd, inner)
    else:
        inner.forward = autocast_fwd
    return model
