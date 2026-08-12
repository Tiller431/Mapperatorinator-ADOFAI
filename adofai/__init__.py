"""
ADOFAI (A Dance of Fire and Ice) format support for Mapperatorinator.

This module provides parsing, event representation, export, and training
capabilities for ADOFAI chart files (.adofai format).
"""

from .event import AdofaiEvent, AdofaiEventType
from .parser import AdofaiLevel, parse_adofai, write_adofai
from .converter import AdofaiConverter

# Dataset and training are optional imports (require torch)
try:
    from .dataset import AdofaiDataset, AdofaiDatasetEntry, collate_adofai_batch
    from .tokenizer import AdofaiTokenizer
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False
    AdofaiDataset = None
    AdofaiDatasetEntry = None
    collate_adofai_batch = None
    AdofaiTokenizer = None

__all__ = [
    'AdofaiEvent',
    'AdofaiEventType',
    'AdofaiLevel',
    'parse_adofai',
    'write_adofai',
    'AdofaiConverter',
    'AdofaiDataset',
    'AdofaiDatasetEntry',
    'collate_adofai_batch',
    'AdofaiTokenizer',
]
