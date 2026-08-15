"""DDP reducer param-set helpers. Torch-only — do not import dataset/slider."""

from __future__ import annotations

import os
from collections import OrderedDict

import torch

_GLOO_GROUP = None


# Child names Mapperatorinator used to declare in incomplete ``__slots__``.
# DDP's ``_verify_param_shape_across_processes`` walks ``named_modules()`` →
# ``_modules`` only. A slot/attribute that is not in ``_modules`` is invisible
# to the reducer, so rank 0 reports 0 params while ranks 1/2/3 still have the
# Muon 194 + AdamW 233 = 427 optimizer-split of the 478,783,248-param model.
MAPPERATORINATOR_CHILD_MODULE_NAMES: tuple[str, ...] = (
    "spectrogram",
    "decoder_embedder",
    "encoder_embedder",
    "transformer",
    "style_embedder",
    "difficulty_embedder",
    "mapper_embedder",
    "song_pos_embedder",
    "loss_fn",
)


def _module_children(model: torch.nn.Module) -> OrderedDict[str, torch.nn.Module]:
    modules = getattr(model, "_modules", None)
    if modules is None:
        return OrderedDict()
    return OrderedDict((name, child) for name, child in modules.items() if child is not None)


def iter_unregistered_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """nn.Module attributes that are not in ``_modules`` (empty-module desync)."""
    registered = _module_children(model)
    found: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()

    names: list[str] = list(MAPPERATORINATOR_CHILD_MODULE_NAMES)
    instance_dict = getattr(model, "__dict__", None)
    if isinstance(instance_dict, dict):
        names.extend(name for name in instance_dict if not name.startswith("_"))
    for name in getattr(type(model), "__slots__", ()):
        if isinstance(name, str):
            names.append(name)

    for name in names:
        if name in registered or name.startswith("_"):
            continue
        try:
            value = getattr(model, name, None)
        except Exception:  # noqa: BLE001 — slot/getattr can raise on a broken tree
            continue
        if not isinstance(value, torch.nn.Module):
            continue
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        found.append((name, value))
    return found


def sync_registered_modules(model: torch.nn.Module) -> list[str]:
    """Put attribute-only children back into ``_modules`` so DDP can see them.

    ``print(model)`` / Muon+AdamW 427 can succeed, then
    ``_verify_param_shape_across_processes`` still see rank 0 as empty if
    ``_modules`` was cleared or never received the slotted children.
    """
    instance_dict = getattr(model, "__dict__", None)
    for state_name in ("_modules", "_parameters", "_buffers"):
        current = getattr(model, state_name, None)
        if current is None:
            current = OrderedDict()
            object.__setattr__(model, state_name, current)
            if isinstance(instance_dict, dict):
                instance_dict[state_name] = current
    restored: list[str] = []
    for name, child in iter_unregistered_modules(model):
        model._modules[name] = child
        restored.append(name)
    return restored


def ddp_reducer_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Parameter tensors DDP actually wraps.

    Matches ``DistributedDataParallel._build_params_for_reducer`` in PyTorch
    2.8: ``named_modules`` + per-module ``named_parameters(recurse=False)``,
    ``requires_grad`` only, skip ``_ddp_params_and_buffers_to_ignore``,
    dedupe shared/tied tensors.

    427 is the Muon 194 + AdamW 233 optimizer-split of the 478,783,248-param
    model, not a tiny net. The 4×A40 hang was NCCL ALLGATHER during DDP
    verify (all 4 ranks together). A 427-vs-0 line *after* that timeout is
    not evidence that rank 0 built an empty module.
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
    """Fail before ``accelerator.prepare`` if this rank's param set is empty."""
    children = _module_children(model)
    try:
        total = sum(1 for _ in model.parameters())
    except AttributeError:
        total = 0
    unregistered = iter_unregistered_modules(model)
    if not children:
        detail = (
            f"unregistered_module_attrs={[name for name, _ in unregistered]!r}"
            if unregistered
            else "no child modules on attributes either"
        )
        raise RuntimeError(
            "DDP _verify_param_shape_across_processes walks named_modules() → "
            "_modules. This rank's module is empty "
            f"(model.parameters() count={total}; {detail}). "
            "The 427 count from the pod is the optimizer split "
            "(Muon 194 + AdamW 233) of the 478,783,248-param model, not a "
            "tiny net. Unfreezing requires_grad does nothing when _modules "
            "has no children."
        )
    if getattr(model, "transformer", None) is not None and children.get("transformer") is None:
        raise RuntimeError(
            "Mapperatorinator.transformer is missing from _modules on this rank. "
            "DDP would treat the module as empty at "
            "_verify_param_shape_across_processes (427-vs-0)."
        )
    signature = ddp_param_signature(model)
    if not signature:
        raise RuntimeError(
            "DDP expects a non-empty requires_grad param set on every rank, but "
            f"this construction has 0 reducer params "
            f"(model.parameters() count={total}, _modules children="
            f"{list(children)!r}). If total==0 the module tree is empty of "
            "Parameter tensors; if total>0 every tensor is frozen."
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


def configure_nccl_for_pcie_multigpu() -> dict[str, str]:
    """Set NCCL defaults *before* ``Accelerator()`` / ``init_process_group``.

    4×A40 field data (main ``a38db9f``, ``num_workers=0``):

    - All 4 ranks printed ``Model loaded`` / ``Embedding(95471, 768)`` first.
    - Hung in DDP ``_verify_param_shape_across_processes`` on the **first**
      collective: ``WorkNCCL SeqNum=1 ALLGATHER NumelIn=1 NumelOut=4``.
    - All 4 ranks timed out together at ~600065–600086 ms (not a straggler).
    - ``427-vs-0`` printed **after** that timeout — garbage from the failed
      ALLGATHER, not a real empty rank-0 module.
    - 1419 MiB / 100% util on all 4; topology PIX/PXB, no NVLink.

    ``accelerate launch`` only auto-sets ``NCCL_P2P_DISABLE`` for RTX 4000.
    A40 is not in that list. Official command is
    ``python osuT5/train.py -cn adofai_v31``. ``NCCL_P2P_LEVEL=NVL`` keeps
    NVLink P2P and disables the PIX/PXB P2P path that hangs the 1-int
    allgather. Do not override an explicit user setting.
    """
    applied: dict[str, str] = {}
    if "NCCL_P2P_DISABLE" not in os.environ and "NCCL_P2P_LEVEL" not in os.environ:
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        applied["NCCL_P2P_LEVEL"] = "NVL"
    if "NCCL_IB_DISABLE" not in os.environ:
        os.environ["NCCL_IB_DISABLE"] = "1"
        applied["NCCL_IB_DISABLE"] = "1"
    return applied


def bind_cuda_device_from_local_rank() -> str | None:
    """Pin this process to its GPU before NCCL init. No-op without LOCAL_RANK."""
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        return None
    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device_index)
    return f"cuda:{device_index}"


def _gloo_process_group():
    """Side Gloo group so param-count gather does not use NCCL."""
    global _GLOO_GROUP
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    if dist.get_backend() == "gloo":
        return dist.group.WORLD
    if _GLOO_GROUP is not None:
        return _GLOO_GROUP
    _GLOO_GROUP = dist.new_group(backend="gloo")
    return _GLOO_GROUP


def allgather_ddp_reducer_counts(count: int) -> list[int]:
    """Gather reducer-tensor counts on Gloo (CPU). Does not touch NCCL.

    The live 4×A40 hang was NCCL ALLGATHER of a 1-int count. This check
    proves every rank has the same 427-split *before* DDP's NCCL verify.
    """
    import torch.distributed as dist

    value = int(count)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    group = _gloo_process_group()
    gathered: list[object | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value, group=group)
    return [int(item) for item in gathered]


def assert_same_ddp_reducer_counts(counts: list[int]) -> None:
    """Every rank must report the same non-zero reducer count (v31: 427)."""
    if not counts:
        raise RuntimeError("DDP reducer-count allgather returned no ranks")
    if any(count <= 0 for count in counts):
        raise RuntimeError(
            "A rank reported 0 DDP reducer tensors before NCCL verify: "
            f"{counts}. This is a real empty/frozen module, not the "
            "post-timeout 427-vs-0 garbage from a hung ALLGATHER."
        )
    if len(set(counts)) != 1:
        raise RuntimeError(
            "Ranks disagree on DDP reducer tensor count before NCCL verify: "
            f"{counts}. Expected one shared split (Muon 194 + AdamW 233 = 427 "
            "on adofai_v31), not a tiny model."
        )


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
