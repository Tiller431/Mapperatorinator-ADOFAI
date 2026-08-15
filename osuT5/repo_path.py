"""Put the repo root on ``sys.path`` for official ``osuT5/train.py`` launches.

``python osuT5/train.py`` and ``accelerate launch osuT5/train.py`` put this
directory first on ``sys.path``. Hydra then chdirs into the run dir, so cwd
is no longer the repo root and ``import adofai`` fails unless the caller set
``PYTHONPATH``.

Insert the repo root *after* the script dir so the inner ``osuT5`` package
(``from osuT5.config``) still wins over the outer ``osuT5/`` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_repo_root_on_sys_path() -> str:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(1 if sys.path else 0, root)
    return root
