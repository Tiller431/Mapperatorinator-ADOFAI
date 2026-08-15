"""Honor train-config mixed_precision through accelerate.

``accelerate launch`` 1.12.0 ``commands/launch.py`` ``_validate_launch_command``
defaults ``--mixed_precision`` to ``'no'`` when the flag is omitted
(``if args.mixed_precision is None: args.mixed_precision = "no"``), then
``utils/launch.py`` writes ``ACCELERATE_MIXED_PRECISION``. ``AcceleratorState``
(``state.py``) uses ``parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")``
when ``Accelerator(mixed_precision=None)``. That silently drops
``adofai_v31`` ``mixed_precision: bf16``.

On 4×A100-80GB the first forward then ran in bf16 (model ``precision: bf16``)
and OOM'd in accelerate ``convert_to_fp32`` / ``tensor.float()``.
"""

from __future__ import annotations

import os
from typing import Any, MutableMapping

_VALID = frozenset({"no", "fp16", "bf16", "fp8"})


def _configured_mixed_precision(args: Any) -> str | None:
    if args is None:
        return None
    if isinstance(args, dict):
        value = args.get("mixed_precision")
    else:
        value = getattr(args, "mixed_precision", None)
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def resolve_mixed_precision(
    args: Any,
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Return the train-config mixed_precision and align the accelerate env.

    Config wins over ``accelerate launch``'s default ``'no'``. An explicit
    config value of ``'no'`` (e.g. ``adofai_whisper_tiny``) is still honored.
    Missing config is an error — do not fall back to the launch default.
    """
    environ = os.environ if environ is None else environ
    configured = _configured_mixed_precision(args)
    if configured is None:
        raise ValueError(
            "train config mixed_precision is missing; refusing accelerate "
            "launch's default 'no' (accelerate 1.12.0 commands/launch.py "
            "_validate_launch_command sets args.mixed_precision='no'; "
            "state.py parse_choice_from_env('ACCELERATE_MIXED_PRECISION', 'no'))"
        )
    if configured not in _VALID:
        raise ValueError(
            f"Unknown mixed_precision {configured!r}. "
            f"Choose from {sorted(_VALID)}"
        )
    environ["ACCELERATE_MIXED_PRECISION"] = configured
    return configured
