"""
Event representation for ADOFAI charts.

ADOFAI events represent the intermediate form between .adofai files and model tokens.
This vocabulary focuses on playable rhythm mechanics (v1 scope).
"""

from __future__ import annotations
import dataclasses
from enum import Enum


class AdofaiEventType(Enum):
    """Event types for ADOFAI chart representation."""
    
    # Timing
    TIME_SHIFT = "time"           # Time in milliseconds
    
    # Tile/angle events
    TILE_ANGLE = "angle"          # Tile angle in degrees (0-359)
    MIDSPIN = "midspin"           # Special midspin tile (999)
    
    # Speed events
    SET_SPEED_BPM = "speed_bpm"   # SetSpeed with BPM
    SET_SPEED_MULT = "speed_mult" # SetSpeed with multiplier
    PAUSE = "pause"               # Pause event
    HOLD = "hold"                 # Hold note duration
    
    # Gameplay modifiers
    TWIRL = "twirl"               # Direction change (twirl)
    MULTI_PLANET = "multiplanet"  # MultiPlanet event
    
    # Metadata/conditioning tokens (used for generation control)
    BPM = "bpm"                   # Initial BPM
    OFFSET = "offset"             # Timing offset in ms
    DIFFICULTY = "difficulty"     # Difficulty rating (if available)
    SONG_LENGTH = "song_length"   # Total song length in ms
    

@dataclasses.dataclass
class AdofaiEvent:
    """
    Represents a single event in the ADOFAI intermediate representation.
    
    Similar to osuT5.event.Event, but for ADOFAI-specific event types.
    """
    type: AdofaiEventType
    value: int | float = 0
    
    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"
    
    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"


@dataclasses.dataclass
class AdofaiEventRange:
    """Defines the valid range for a quantized event type."""
    type: AdofaiEventType
    min_value: int
    max_value: int
