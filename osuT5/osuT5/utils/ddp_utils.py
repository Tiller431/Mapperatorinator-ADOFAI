"""DDP reducer param-set helpers. Torch-only — do not import dataset/slider."""

from __future__ import annotations

import torch


def ddp_reducer_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Parameter tensors DDP actually wraps.

    Matches ``DistributedDataParallel._build_params_for_reducer`` in PyTorch
    2.8: ``named_modules`` + per-module ``named_parameters(recurse=False)``,
    ``requires_grad`` only, skip ``_ddp_params_and_buffers_to_ignore``,
    dedupe shared/tied tensors.

    ``model.parameters()`` / ``print_model_parameters`` count *all* tensors
    (including frozen). A rank can print Embedding 95471×768 and 478.8M
    total weights, then DDP still report **0 params** if nothing requires
    grad — the v31 4×A40 failure mode.
    """
    ignore = set(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
    seen: set[int] = set()
    out: list[tuple[str, torch.nn.Parameter]] = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fqn = f"{module_name}.{param_name}" if module_name else param_name
            if not param.requires_grad or fqn in ignore:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            out.append((fqn, param))
    return out


def ddp_param_signature(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Stable (name, shape) set used to compare two independent constructions."""
    return tuple((name, tuple(param.shape)) for name, param in ddp_reducer_named_parameters(model))


def assert_model_ready_for_ddp(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Fail before ``accelerator.prepare`` if this rank would hand DDP 0 params."""
    signature = ddp_param_signature(model)
    total = sum(1 for _ in model.parameters())
    if not signature:
        raise RuntimeError(
            "DDP expects a non-empty requires_grad param set on every rank, but "
            f"this construction has 0 reducer params "
            f"(model.parameters() count={total}, including frozen). "
            "Rank 0 printing 478.8M total weights is not enough — the reducer "
            "only sees requires_grad=True tensors. Typical causes: wandb / "
            "inference_mode leftover on the main process, meta-device empty "
            "modules, or a rank-only model build."
        )
    meta = [
        name
        for name, param in ddp_reducer_named_parameters(model)
        if getattr(param, "device", None) is not None and param.device.type == "meta"
    ]
    if meta:
        raise RuntimeError(
            "DDP cannot wrap meta-device parameters "
            f"(empty module on this rank): {meta[:8]}"
        )
    return signature


def assert_identical_ddp_param_sets(
        signature_a: tuple[tuple[str, tuple[int, ...]], ...],
        signature_b: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    """Two independent builds (simulated ranks) must agree on name+shape."""
    if signature_a != signature_b:
        names_a = {name for name, _ in signature_a}
        names_b = {name for name, _ in signature_b}
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        raise AssertionError(
            "Independent model constructions produced different DDP param sets "
            f"(len {len(signature_a)} vs {len(signature_b)}). "
            f"only_first={only_a[:12]!r} only_second={only_b[:12]!r}"
        )
