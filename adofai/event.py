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
    
    # Gameplay modifiers (MUST-HAVE)
    TWIRL = "twirl"                     # Direction change (reverse)
    MULTI_PLANET = "multiplanet"        # MultiPlanet (2 vs 3+)
    CHECKPOINT = "checkpoint"           # Progress checkpoint
    AUTO_PLAY_TILES = "autoplay"        # AutoPlayTiles
    SET_PLANET_ROTATION = "planet_rot"  # SetPlanetRotation
    FREE_ROAM = "freeroam"              # FreeRoam mode
    FREE_ROAM_TWIRL = "freeroam_twirl"  # FreeRoamTwirl
    FREE_ROAM_REMOVE = "freeroam_remove"  # FreeRoamRemove
    SCALE_MARGIN = "scale_margin"       # ScaleMargin
    SCALE_RADIUS = "scale_radius"       # ScaleRadius
    MULTITAP = "multitap"               # Multitap
    HIDE = "hide"                       # Hide event
    KILL_PLAYER = "kill_player"         # KillPlayer
    
    # Track events (MUST-HAVE)
    POSITION_TRACK = "position_track"   # PositionTrack
    MOVE_TRACK = "move_track"           # MoveTrack
    COLOR_TRACK = "color_track"         # ColorTrack
    ANIMATE_TRACK = "animate_track"     # AnimateTrack
    MOVE_CAMERA = "move_camera"         # MoveCamera
    
    # Audio events (MUST-HAVE)
    SET_HITSOUND = "set_hitsound"       # SetHitsound
    PLAY_SOUND = "play_sound"           # PlaySound
    SET_HOLD_SOUND = "set_hold_sound"   # SetHoldSound
    
    # Control flow events (MUST-HAVE)
    REPEAT_EVENTS = "repeat_events"     # RepeatEvents
    SET_CONDITIONAL_EVENTS = "set_cond" # SetConditionalEvents
    SET_INPUT_EVENT = "set_input"       # SetInputEvent
    
    # VFX events (Tyler override - keep these)
    FLASH = "flash"                     # Flash effect
    BLOOM = "bloom"                     # Bloom lighting
    SHAKE_SCREEN = "shake_screen"       # ShakeScreen
    SET_FILTER = "set_filter"           # SetFilter
    
    # Event parameters (quantized sub-events for camera/VFX)
    CAMERA_POSITION_X = "cam_pos_x"     # Camera/track X position (quantized)
    CAMERA_POSITION_Y = "cam_pos_y"     # Camera/track Y position (quantized)
    CAMERA_ZOOM = "cam_zoom"            # Zoom level (quantized)
    CAMERA_ROTATION = "cam_rotation"    # Rotation angle (quantized)
    CAMERA_DURATION = "cam_duration"    # Duration in beats (quantized)
    CAMERA_EASE = "cam_ease"            # Easing type (enum)
    CAMERA_RELATIVE = "cam_relative"    # RelativeTo enum (Player/Tile/Global/etc)
    
    # VFX parameters
    COLOR_RGB = "color_rgb"             # Color value (quantized RGB)
    OPACITY = "opacity"                 # Opacity percentage
    INTENSITY = "intensity"             # Effect intensity
    FILTER_TYPE = "filter_type"         # Filter type enum
    
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
