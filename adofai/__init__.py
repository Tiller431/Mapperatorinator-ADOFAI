"""
ADOFAI (A Dance of Fire and Ice) format support for Mapperatorinator.

This module provides parsing, event representation, and export capabilities
for ADOFAI chart files (.adofai format).
"""

from .event import AdofaiEvent, AdofaiEventType
from .parser import AdofaiLevel, parse_adofai, write_adofai
from .converter import AdofaiConverter

__all__ = [
    'AdofaiEvent',
    'AdofaiEventType',
    'AdofaiLevel',
    'parse_adofai',
    'write_adofai',
    'AdofaiConverter',
]
