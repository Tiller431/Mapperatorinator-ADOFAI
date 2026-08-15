"""adofai_v31 mixed_precision: bf16 must reach Accelerator.

Field data (main ``2e971b3``, 4×A100-80GB, batch 8, compile=false, tgt 8192):

- DDP/NCCL verify passed (427-vs-427, 478.8M).
- First forward OOM: tried 17.48 GiB with 9.21 GiB free (70.03/79.25 used).
- Forward produced bf16 tensors (``precision: bf16``). OOM was accelerate
  ``convert_to_fp32`` / ``tensor.float()`` after that.

Cause: ``accelerate launch`` 1.12.0 ``commands/launch.py`` (``_validate_launch_command``)
defaults ``--mixed_precision`` to ``'no'`` when the flag is omitted
(warning: "was set to a value of ``'no'``"), then writes
``ACCELERATE_MIXED_PRECISION``. ``AcceleratorState``
(``state.py``) uses ``parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")``
when the constructor argument is None. That silently drops
``configs/train/adofai_v31.yaml`` ``mixed_precision: bf16``.

These tests must fail if that default wins.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_mixed_precision():
    """Load the torch-free helper without importing osuT5.utils (slider)."""
    import importlib.util

    path = REPO_ROOT / "osuT5" / "osuT5" / "utils" / "mixed_precision.py"
    spec = importlib.util.spec_from_file_location("osuT5_mixed_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adofai_v31_train_args():
    """adofai_v31.yaml values train.py would pass into resolve_mixed_precision.

    Full Hydra compose needs ConfigStore ``train/base`` and a specific
    ``sys.path`` layout; other tests mutate that. The yaml is the config
    ``-cn adofai_v31`` loads (overrides ``default.yaml``).
    """
    default = yaml.safe_load(
        (REPO_ROOT / "configs" / "train" / "default.yaml").read_text(encoding="utf-8")
    )
    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "train" / "adofai_v31.yaml").read_text(encoding="utf-8")
    )
    return SimpleNamespace(
        mixed_precision=data.get("mixed_precision", default.get("mixed_precision")),
        precision=data.get("precision", default.get("precision")),
        device=data.get("device", default.get("device", "gpu")),
    )


def _reset_accelerate_state():
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()


def test_adofai_v31_yaml_mixed_precision_is_bf16_and_width_unchanged():
    data = yaml.safe_load(
        (REPO_ROOT / "configs" / "train" / "adofai_v31.yaml").read_text(encoding="utf-8")
    )
    assert data["mixed_precision"] == "bf16"
    assert data["precision"] == "bf16"
    assert data["model"]["cond_size"] == 128
    assert data["data"]["tgt_seq_len"] == 8192
    assert data["optim"]["batch_size"] == 32


def test_train_py_resolves_mixed_precision_before_accelerator():
    """Must not pass a raw/None value that AcceleratorState defaults to 'no'."""
    src = (REPO_ROOT / "osuT5" / "train.py").read_text(encoding="utf-8")
    assert "resolve_mixed_precision" in src
    assert src.index("resolve_mixed_precision") < src.index("accelerator = Accelerator(")
    assert "python osuT5/train.py -cn adofai_v31" in (
        REPO_ROOT / "adofai" / "train.py"
    ).read_text(encoding="utf-8")


def test_launch_default_no_does_not_override_v31_bf16(monkeypatch):
    """accelerate launch sets ACCELERATE_MIXED_PRECISION=no; config must win."""
    mp = _load_mixed_precision()
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    resolved = mp.resolve_mixed_precision(SimpleNamespace(mixed_precision="bf16"))
    assert resolved == "bf16"
    assert os.environ["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_missing_mixed_precision_does_not_default_to_no(monkeypatch):
    """Silent drop to 'no' is the OOM bug. Refuse it."""
    mp = _load_mixed_precision()
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    with pytest.raises(ValueError, match="no"):
        mp.resolve_mixed_precision(SimpleNamespace())


def test_explicit_config_no_is_still_honored(monkeypatch):
    """adofai_whisper_tiny sets mixed_precision: 'no' on purpose."""
    mp = _load_mixed_precision()
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)
    assert mp.resolve_mixed_precision(SimpleNamespace(mixed_precision="no")) == "no"


def test_adofai_v31_config_resolves_bf16_when_launch_defaults_to_no(monkeypatch):
    mp = _load_mixed_precision()
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    args = _adofai_v31_train_args()
    assert args.mixed_precision == "bf16"
    assert mp.resolve_mixed_precision(args) == "bf16"
    assert os.environ["ACCELERATE_MIXED_PRECISION"] == "bf16"


def test_accelerator_from_adofai_v31_uses_bf16_without_gpus(monkeypatch):
    """Construct Accelerator from adofai_v31. Must be bf16 even if launch said no."""
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    pytest.importorskip("hydra")
    mp = _load_mixed_precision()

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_state()

    args = _adofai_v31_train_args()
    mixed_precision = mp.resolve_mixed_precision(args)
    assert mixed_precision == "bf16"

    accelerator = accelerate.Accelerator(cpu=True, mixed_precision=mixed_precision)
    try:
        assert accelerator.mixed_precision == "bf16"
        assert os.environ["ACCELERATE_MIXED_PRECISION"] == "bf16"
    finally:
        _reset_accelerate_state()


def test_accelerator_none_mixed_precision_follows_launch_default_no(monkeypatch):
    """Documents the drop: constructor None + env 'no' => 'no'. Our path must not do this."""
    pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")

    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_state()
    try:
        dropped = accelerate.Accelerator(cpu=True, mixed_precision=None)
        assert dropped.mixed_precision == "no"
    finally:
        _reset_accelerate_state()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
