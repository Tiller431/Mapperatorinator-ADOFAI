"""adofai.converter must import under both sys.path layouts.

``python osuT5/train.py`` puts the osuT5/ directory first on sys.path, so the
inner package is named ``osuT5`` (``from osuT5.event``).

``python inference.py`` puts the repo root first, so the inner package is
named ``osuT5.osuT5`` (``from osuT5.osuT5.event``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OSUT5_DIR = REPO_ROOT / "osuT5"


def _purge_import_cache() -> dict:
    saved = {}
    for name in list(sys.modules):
        if name == "osuT5" or name.startswith("osuT5.") or name == "adofai" or name.startswith("adofai."):
            saved[name] = sys.modules.pop(name)
    return saved


def _import_converter(sys_path_head: list[str]):
    saved_path = sys.path[:]
    saved_modules = _purge_import_cache()
    try:
        sys.path = list(sys_path_head) + [p for p in saved_path if p not in sys_path_head]
        import adofai.converter as converter

        assert converter.Event is not None
        assert converter.EventType is not None
        assert converter.AdofaiConverter is not None
        assert converter.EventType.TILE_ANGLE is not None
        return converter
    finally:
        _purge_import_cache()
        sys.path[:] = saved_path
        sys.modules.update(saved_modules)


def test_converter_imports_with_repo_root_on_sys_path():
    """Repo-root generate path: python inference.py -cn adofai_v31."""
    _import_converter([str(REPO_ROOT)])


def test_converter_imports_with_osut5_dir_on_sys_path():
    """Trainer smoke path: python osuT5/train.py -cn adofai_v31.

    Script dir (osuT5/) is first; repo root stays on the path so ``adofai``
    resolves. This is the layout that raised ``No module named osuT5.osuT5``.
    """
    _import_converter([str(OSUT5_DIR), str(REPO_ROOT)])


if __name__ == "__main__":
    test_converter_imports_with_repo_root_on_sys_path()
    print("repo-root converter import: ok")
    test_converter_imports_with_osut5_dir_on_sys_path()
    print("osuT5/ converter import: ok")
