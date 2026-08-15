"""
ADOFAI (A Dance of Fire and Ice) I/O for Mapperatorinator.

Production train/infer path is osuT5 (Event / EventRange / Tokenizer).
This package keeps parser + converter as I/O only.
"""

from .parser import AdofaiLevel, parse_adofai, write_adofai
from .converter import AdofaiConverter
from .export import events_to_adofai_file

__all__ = [
    "AdofaiLevel",
    "parse_adofai",
    "write_adofai",
    "AdofaiConverter",
    "events_to_adofai_file",
]
