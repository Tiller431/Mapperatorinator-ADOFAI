"""Chunked linear + CE so train never materializes full [B, T, V] logits.

``proj_out`` / ``lm_head`` on decoder hidden ``[B, T, D]`` would allocate
``[B, T, V]``. For v31 that is 8 × 8192 × 95471 × 2 bytes = 11.65 GiB, which
is the first-step OOM at ``loss_fn(swapaxes(output.logits, 1, -1), labels)``.

Apply the head in chunks over T. Checkpoint each chunk so backward recomputes
``[B, C, V]`` instead of keeping every chunk's logits.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LABEL_IGNORE_ID = -100
DEFAULT_CHUNK_SIZE = 128

# RoPEWhisper / NWhisper / Whisper ``.model`` forward kwargs. Extra Mapperatorinator
# keys (frames, labels, sample_weights, …) must not reach the body.
_BACKBONE_MODEL_KEYS = frozenset(
    {
        "input_features",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "head_mask",
        "decoder_head_mask",
        "cross_attn_head_mask",
        "encoder_outputs",
        "past_key_values",
        "decoder_inputs_embeds",
        "decoder_position_ids",
        "use_cache",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "cache_position",
        "inputs_embeds",
        "input_values",
    }
)


def decoder_hidden_states(transformer: Any, inputs: dict) -> torch.Tensor:
    """Encoder-decoder body only. Do not run ``proj_out`` / ``lm_head``."""
    body = getattr(transformer, "model", None)
    if body is None:
        raise RuntimeError(
            "Train CE needs transformer.model (decoder hidden states). "
            "transformer.forward applies proj_out/lm_head and materializes "
            "[B, T, V] logits."
        )
    kwargs = {key: value for key, value in inputs.items() if key in _BACKBONE_MODEL_KEYS}
    kwargs["use_cache"] = False
    outputs = body(**kwargs)
    return outputs[0]


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    lm_head: Any,
    labels: torch.Tensor,
    *,
    class_weight: Optional[torch.Tensor] = None,
    ignore_index: int = LABEL_IGNORE_ID,
    label_smoothing: float = 0.0,
    sample_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    logit_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CE(lm_head(hidden), labels) without a full ``[B, T, V]`` logit tensor.

    ``hidden`` is ``[B, T, D]``. ``lm_head`` is the output embedding / Linear
    (``proj_out``). Reduction matches Mapperatorinator: mean over non-ignored
    tokens, optional per-sample weights.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [B, T, D], got {tuple(hidden.shape)}")
    if labels.shape[:2] != hidden.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} must match hidden {tuple(hidden.shape)} on B,T"
        )

    weight = lm_head.weight if hasattr(lm_head, "weight") else lm_head
    bias = getattr(lm_head, "bias", None) if hasattr(lm_head, "weight") else None
    has_bias = bias is not None
    has_class_weight = class_weight is not None
    has_scale = logit_scale is not None

    # Checkpoint requires tensor args. Unused optionals are dummy scalars.
    if bias is None:
        bias = weight.new_zeros(())
    if class_weight is None:
        class_weight = weight.new_zeros(0)
    if logit_scale is None:
        logit_scale = weight.new_ones(())

    def _ce_chunk(hidden_chunk, labels_chunk, weight_t, bias_t, class_weight_t, scale_t):
        logits = F.linear(hidden_chunk, weight_t, bias_t if has_bias else None)
        if has_scale:
            logits = logits * scale_t
        return F.cross_entropy(
            logits.transpose(1, 2),
            labels_chunk,
            weight=class_weight_t if has_class_weight else None,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    _, tgt_len, _ = hidden.shape
    pieces = []
    use_ckpt = bool(hidden.requires_grad and torch.is_grad_enabled())
    for start in range(0, tgt_len, chunk_size):
        end = min(start + chunk_size, tgt_len)
        args = (
            hidden[:, start:end],
            labels[:, start:end],
            weight,
            bias,
            class_weight,
            logit_scale,
        )
        if use_ckpt:
            piece = checkpoint(_ce_chunk, *args, use_reentrant=False)
        else:
            piece = _ce_chunk(*args)
        pieces.append(piece)

    unreduced = torch.cat(pieces, dim=1)
    if sample_weights is not None:
        unreduced = unreduced * sample_weights.unsqueeze(1)
    n_valid = (labels != ignore_index).sum()
    return unreduced.sum() / n_valid
