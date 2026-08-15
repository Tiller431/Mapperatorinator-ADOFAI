"""
Converter between ADOFAI events and model tokens.

This module bridges ADOFAI's event representation and the tokenizer vocabulary,
handling conversion to/from the intermediate event format used by the model.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

try:
    from osuT5.event import Event, EventType
except ModuleNotFoundError:
    from osuT5.osuT5.event import Event, EventType
from .parser import AdofaiLevel


class AdofaiConverter:
    """
    Converts between ADOFAI levels and event sequences for model training/inference.
    
    Handles:
    - Converting .adofai structure to time-ordered events
    - Converting event sequences back to .adofai structure
    - Timing calculations (angle rotations, BPM, twirls)
    """
    
    def __init__(self):
        """Initialize converter with default settings."""
        self.time_precision = 10  # Time quantization in ms (match osu! convention)
        
        # Camera ease types (common ADOFAI easing functions)
        self.ease_types = [
            "Linear", "InSine", "OutSine", "InOutSine",
            "InQuad", "OutQuad", "InOutQuad",
            "InCubic", "OutCubic", "InOutCubic",
            "InQuart", "OutQuart", "InOutQuart",
            "InQuint", "OutQuint", "InOutQuint",
            "InExpo", "OutExpo", "InOutExpo",
            "InCirc", "OutCirc", "InOutCirc",
            "InBack", "OutBack", "InOutBack",
            "InElastic", "OutElastic", "InOutElastic",
            "InBounce", "OutBounce", "InOutBounce"
        ]
        
        # RelativeTo enum values
        self.relative_to_types = [
            "Player", "Tile", "Global", "LastPosition", "LastPositionNoRotation"
        ]
        # MagicShaper SetFilter.filter enum (NOT ADOFAI-JS Pixellate/Bloom/Warp/RadialBlur/Custom)
        self.filter_types = [
            "None", "Grayscale", "Sepia", "Invert", "VHS",
            "EightiesTV", "FiftiesTV", "Arcade", "LED", "Rain",
            "Blizzard", "PixelSnow", "Compression", "Glitch", "Pixelate",
            "Waves", "Static", "Grain", "MotionBlur", "Fisheye",
            "Aberration", "Drawing", "Neon", "Handheld", "NightVision",
            "Funk", "Tunnel", "Weird3D", "Blur", "BlurFocus",
            "GaussianBlur", "HexagonBlack", "Posterize", "Sharpen", "Contrast",
            "EdgeBlackLine", "OilPaint", "SuperDot", "WaterDrop", "LightWater",
            "Petals", "PetalsInstant",
        ]
        self.tile_anchors = ["ThisTile", "Start", "End"]
    
    def _ease_to_id(self, ease: str) -> int:
        """Convert ease type string to ID."""
        try:
            return self.ease_types.index(ease)
        except ValueError:
            return 0  # Default to Linear
    
    def _id_to_ease(self, ease_id: int) -> str:
        """Convert ease ID to string."""
        if 0 <= ease_id < len(self.ease_types):
            return self.ease_types[ease_id]
        return "Linear"
    
    def _relative_to_id(self, relative: str) -> int:
        """Convert relativeTo string to ID."""
        try:
            return self.relative_to_types.index(relative)
        except ValueError:
            return 0  # Default to Player
    
    def _id_to_relative(self, relative_id: int) -> str:
        """Convert relativeTo ID to string."""
        if 0 <= relative_id < len(self.relative_to_types):
            return self.relative_to_types[relative_id]
        return "Player"

    def _filter_to_id(self, name: str) -> int:
        try:
            return self.filter_types.index(name)
        except ValueError:
            return 0

    def _id_to_filter(self, filter_id: int) -> str:
        if 0 <= filter_id < len(self.filter_types):
            return self.filter_types[filter_id]
        return "None"

    def _enabled_to_id(self, value) -> int:
        if isinstance(value, str):
            return 1 if value == "Enabled" else 0
        return 1 if value else 0

    def _id_to_enabled(self, value: int) -> str:
        return "Enabled" if value else "Disabled"

    def _hex_to_color_id(self, hex_color: str) -> int:
        text = str(hex_color).lstrip("#")
        if len(text) < 6:
            return 0
        try:
            r = int(text[0:2], 16) >> 4
            g = int(text[2:4], 16) >> 4
            b = int(text[4:6], 16) >> 4
            return (r << 8) | (g << 4) | b
        except ValueError:
            return 0

    def _color_id_to_hex(self, color_id: int) -> str:
        r = ((int(color_id) >> 8) & 0xF) * 17
        g = ((int(color_id) >> 4) & 0xF) * 17
        b = (int(color_id) & 0xF) * 17
        return f"{r:02x}{g:02x}{b:02x}"

    def _tile_ref_to_id(self, ref) -> int:
        if not isinstance(ref, (list, tuple)) or len(ref) < 2:
            return 64 * 4
        offset = max(-64, min(64, int(ref[0])))
        try:
            anchor = self.tile_anchors.index(ref[1])
        except ValueError:
            anchor = 0
        return (offset + 64) * 4 + anchor

    def _id_to_tile_ref(self, packed: int) -> list:
        packed = int(packed)
        anchor = packed % 4
        offset = packed // 4 - 64
        name = self.tile_anchors[anchor] if 0 <= anchor < len(self.tile_anchors) else "ThisTile"
        return [offset, name]
    
    def level_to_events(self, level: AdofaiLevel) -> tuple[list[Event], list[float]]:
        """
        Convert ADOFAI level to time-ordered event sequence.
        
        Args:
            level: Parsed ADOFAI level
            
        Returns:
            Tuple of (events, event_times) where:
                - events: List of osuT5 Event objects
                - event_times: List of timestamps in milliseconds
        """
        events = []
        event_times = []
        
        # Extract initial settings
        settings = level.settings
        initial_bpm = settings.get("bpm", 100)
        offset = settings.get("offset", 0)
        
        # Add metadata events at the start
        events.append(Event(EventType.BPM, int(initial_bpm)))
        event_times.append(0)
        
        events.append(Event(EventType.OFFSET, int(offset)))
        event_times.append(0)

        pitch = settings.get("pitch", 100)
        events.append(Event(EventType.PITCH, int(pitch)))
        event_times.append(0)
        
        # Add difficulty if available
        difficulty = settings.get("difficulty", None)
        if difficulty is not None:
            events.append(Event(EventType.DIFFICULTY, int(difficulty)))
            event_times.append(0)
        
        # Track state for timing calculations
        current_time = offset
        current_bpm = initial_bpm
        current_heading = 180.0  # Game starts heading 180° (not 0°)
        twirl_clockwise = True  # Default rotation direction
        hold_duration = 0.0  # Accumulated hold duration for next tile
        multiplanet_count = 2  # Game default is two planets; MultiPlanet persists
        pending_pause_ms = 0.0  # Pause waits on the current tile, then travel continues
        
        # Build floor-to-actions mapping
        floor_actions = {}
        for action in level.actions:
            floor = action.get("floor", 0)
            if floor not in floor_actions:
                floor_actions[floor] = []
            floor_actions[floor].append(action)
        
        # Process each tile
        for floor_idx, tile_angle in enumerate(level.angle_data):
            # Process actions for this floor first (they affect timing)
            if floor_idx in floor_actions:
                for action in floor_actions[floor_idx]:
                    event_type = action.get("eventType", "")
                    
                    if event_type == "SetSpeed":
                        speed_type = action.get("speedType", "Bpm")
                        if speed_type == "Bpm":
                            new_bpm = action.get("beatsPerMinute", current_bpm)
                            events.append(Event(EventType.SET_SPEED_BPM, int(new_bpm)))
                            event_times.append(current_time)
                            current_bpm = new_bpm
                        elif speed_type == "Multiplier":
                            multiplier = float(action.get("bpmMultiplier", 1.0))
                            events.append(Event(EventType.SET_SPEED_MULT, int(round(multiplier * 10))))
                            event_times.append(current_time)
                            current_bpm *= multiplier
                        events.append(Event(EventType.ANGLE_OFFSET, int(action.get("angleOffset", 0))))
                        event_times.append(current_time)
                    
                    elif event_type == "Twirl":
                        twirl_clockwise = not twirl_clockwise
                        events.append(Event(EventType.TWIRL, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "Pause":
                        duration = float(action.get("duration", 1.0))
                        events.append(Event(EventType.PAUSE, int(round(duration * 10))))
                        event_times.append(current_time)
                        events.append(Event(EventType.PAUSE_COUNTDOWN, int(action.get("countdownTicks", 0))))
                        event_times.append(current_time)
                        angle_dir = int(action.get("angleCorrectionDir", -1))
                        events.append(Event(EventType.PAUSE_ANGLE_DIR, { -1: 0, 0: 1, 1: 2 }.get(angle_dir, 0)))
                        event_times.append(current_time)
                        pending_pause_ms += duration * (60000.0 / current_bpm)
                    
                    elif event_type == "Hold":
                        duration = float(action.get("duration", 1.0))
                        events.append(Event(EventType.HOLD, int(round(duration * 10))))
                        event_times.append(current_time)
                        events.append(Event(EventType.HOLD_DISTANCE, int(action.get("distanceMultiplier", 100))))
                        event_times.append(current_time)
                        landing = action.get("landingAnimation", False)
                        events.append(Event(EventType.HOLD_LANDING, self._enabled_to_id(landing) if not isinstance(landing, bool) else int(landing)))
                        event_times.append(current_time)
                        hold_duration += duration
                    
                    elif event_type == "MultiPlanet":
                        planets_raw = action.get("planets", "TwoPlanets")
                        if planets_raw in (3, "ThreePlanets"):
                            planets = 3
                        else:
                            planets = 2
                        events.append(Event(EventType.MULTI_PLANET, planets))
                        event_times.append(current_time)
                        multiplanet_count = planets
                    
                    elif event_type == "MoveCamera":
                        # Camera movement: emit main event + parameters
                        events.append(Event(EventType.MOVE_CAMERA, 1))
                        event_times.append(current_time)
                        
                        # Position (quantized to grid, default 0,0)
                        pos = action.get("position", [0, 0])
                        events.append(Event(EventType.CAMERA_POSITION_X, int(pos[0])))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_POSITION_Y, int(pos[1])))
                        event_times.append(current_time)
                        
                        # Rotation (quantized to degrees)
                        rotation = action.get("rotation", 0)
                        events.append(Event(EventType.CAMERA_ROTATION, int(rotation)))
                        event_times.append(current_time)
                        
                        # Zoom (quantized to percentage)
                        zoom = action.get("zoom", 100)
                        events.append(Event(EventType.CAMERA_ZOOM, int(zoom)))
                        event_times.append(current_time)
                        
                        # Duration in beats, quantized to 0.1-beat steps
                        duration = action.get("duration", 1.0)
                        events.append(Event(EventType.CAMERA_DURATION, int(round(float(duration) * 10))))
                        event_times.append(current_time)
                        
                        # Ease type (string -> enum ID)
                        ease = action.get("ease", "Linear")
                        ease_id = self._ease_to_id(ease)
                        events.append(Event(EventType.CAMERA_EASE, ease_id))
                        event_times.append(current_time)
                        
                        # RelativeTo (string -> enum ID)
                        relative_to = action.get("relativeTo", "Player")
                        relative_id = self._relative_to_id(relative_to)
                        events.append(Event(EventType.CAMERA_RELATIVE, relative_id))
                        event_times.append(current_time)
                        events.append(Event(EventType.ANGLE_OFFSET, int(action.get("angleOffset", 0))))
                        event_times.append(current_time)
                    
                    elif event_type == "MoveTrack":
                        events.append(Event(EventType.MOVE_TRACK, 1))
                        event_times.append(current_time)
                        pos_offset = action.get("positionOffset", [0, 0])
                        events.append(Event(EventType.CAMERA_POSITION_X, int(pos_offset[0])))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_POSITION_Y, int(pos_offset[1])))
                        event_times.append(current_time)
                        events.append(Event(EventType.TRACK_START_TILE, self._tile_ref_to_id(action.get("startTile", [0, "ThisTile"]))))
                        event_times.append(current_time)
                        events.append(Event(EventType.TRACK_END_TILE, self._tile_ref_to_id(action.get("endTile", [0, "ThisTile"]))))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_DURATION, int(round(float(action.get("duration", 1.0)) * 10))))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_EASE, self._ease_to_id(action.get("ease", "Linear"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.ANGLE_OFFSET, int(action.get("angleOffset", 0))))
                        event_times.append(current_time)
                    
                    # Must-have gameplay events
                    elif event_type == "Checkpoint":
                        events.append(Event(EventType.CHECKPOINT, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "AutoPlayTiles":
                        enabled = 1 if action.get("enabled", True) else 0
                        events.append(Event(EventType.AUTO_PLAY_TILES, enabled))
                        event_times.append(current_time)
                    
                    elif event_type == "SetPlanetRotation":
                        ease_parts = action.get("easeParts", 1)
                        events.append(Event(EventType.SET_PLANET_ROTATION, ease_parts))
                        event_times.append(current_time)
                    
                    elif event_type == "FreeRoam":
                        events.append(Event(EventType.FREE_ROAM, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "FreeRoamTwirl":
                        events.append(Event(EventType.FREE_ROAM_TWIRL, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "FreeRoamRemove":
                        events.append(Event(EventType.FREE_ROAM_REMOVE, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "ScaleMargin":
                        scale = action.get("scale", 100)
                        events.append(Event(EventType.SCALE_MARGIN, int(scale)))
                        event_times.append(current_time)
                    
                    elif event_type == "ScaleRadius":
                        scale = action.get("scale", 100)
                        events.append(Event(EventType.SCALE_RADIUS, int(scale)))
                        event_times.append(current_time)
                    
                    elif event_type == "Multitap":
                        presses = action.get("presses", 2)
                        events.append(Event(EventType.MULTITAP, presses))
                        event_times.append(current_time)
                    
                    elif event_type == "Hide":
                        hide_judge = 1 if action.get("hideJudgment", False) else 0
                        events.append(Event(EventType.HIDE, hide_judge))
                        event_times.append(current_time)
                    
                    elif event_type == "KillPlayer":
                        events.append(Event(EventType.KILL_PLAYER, 1))
                        event_times.append(current_time)
                    
                    # Track events
                    elif event_type == "PositionTrack":
                        pos = action.get("position", [0, 0])
                        events.append(Event(EventType.POSITION_TRACK, 1))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_POSITION_X, int(pos[0])))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_POSITION_Y, int(pos[1])))
                        event_times.append(current_time)
                    
                    elif event_type == "ColorTrack":
                        track_color_type = action.get("trackColorType", "Single")
                        # Simplified: emit type as integer
                        color_type_id = 0 if track_color_type == "Single" else 1
                        events.append(Event(EventType.COLOR_TRACK, color_type_id))
                        event_times.append(current_time)
                    
                    elif event_type == "AnimateTrack":
                        track_anim = action.get("trackAnimation", "None")
                        # Simplified: emit as binary flag
                        anim_id = 0 if track_anim == "None" else 1
                        events.append(Event(EventType.ANIMATE_TRACK, anim_id))
                        event_times.append(current_time)
                    
                    # Audio events
                    elif event_type == "SetHitsound":
                        hitsound = action.get("hitsound", "Kick")
                        # Map common hitsounds to IDs (simplified)
                        hitsound_map = {"Kick": 0, "Snare": 1, "Hat": 2, "Shaker": 3, "Sizzle": 4}
                        hitsound_id = hitsound_map.get(hitsound, 0)
                        events.append(Event(EventType.SET_HITSOUND, hitsound_id))
                        event_times.append(current_time)
                    
                    elif event_type == "PlaySound":
                        events.append(Event(EventType.PLAY_SOUND, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "SetHoldSound":
                        hitsound = action.get("hitsound", "Kick")
                        hitsound_map = {"Kick": 0, "Snare": 1, "Hat": 2, "Shaker": 3, "Sizzle": 4}
                        hitsound_id = hitsound_map.get(hitsound, 0)
                        events.append(Event(EventType.SET_HOLD_SOUND, hitsound_id))
                        event_times.append(current_time)
                    
                    # Control flow events
                    elif event_type == "RepeatEvents":
                        repetitions = action.get("repetitions", 1)
                        events.append(Event(EventType.REPEAT_EVENTS, repetitions))
                        event_times.append(current_time)
                    
                    elif event_type == "SetConditionalEvents":
                        events.append(Event(EventType.SET_CONDITIONAL_EVENTS, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "SetInputEvent":
                        events.append(Event(EventType.SET_INPUT_EVENT, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "Flash":
                        events.append(Event(EventType.FLASH, int(round(float(action.get("duration", 1.0)) * 10))))
                        event_times.append(current_time)
                        plane = 0 if action.get("plane", "Foreground") == "Background" else 1
                        events.append(Event(EventType.VFX_PLANE, plane))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_COLOR, self._hex_to_color_id(action.get("startColor", "ffffff"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_OPACITY, int(action.get("startOpacity", 100))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_COLOR, self._hex_to_color_id(action.get("endColor", "ffffff"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_OPACITY, int(action.get("endOpacity", 0))))
                        event_times.append(current_time)
                        events.append(Event(EventType.ANGLE_OFFSET, int(action.get("angleOffset", 0))))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_EASE, self._ease_to_id(action.get("ease", "Linear"))))
                        event_times.append(current_time)
                    
                    elif event_type == "Bloom":
                        enabled = self._enabled_to_id(action.get("enabled", "Enabled"))
                        intensity = int(action.get("intensity", 100))
                        events.append(Event(EventType.BLOOM, intensity if enabled else 0))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_ENABLED, enabled))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_THRESHOLD, int(action.get("threshold", 50))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_COLOR, self._hex_to_color_id(action.get("color", "ffffff"))))
                        event_times.append(current_time)
                    
                    elif event_type == "ShakeScreen":
                        events.append(Event(EventType.SHAKE_SCREEN, int(action.get("intensity", 100))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_STRENGTH, int(action.get("strength", 100))))
                        event_times.append(current_time)
                    
                    elif event_type == "SetFilter":
                        events.append(Event(EventType.SET_FILTER, self._filter_to_id(action.get("filter", "None"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_ENABLED, self._enabled_to_id(action.get("enabled", "Enabled"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_INTENSITY, int(action.get("intensity", 100))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_DISABLE_OTHERS, self._enabled_to_id(action.get("disableOthers", False))))
                        event_times.append(current_time)
                    
                    elif event_type == "SetFilterAdvanced":
                        events.append(Event(EventType.SET_FILTER_ADVANCED, self._filter_to_id(action.get("filter", "None"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_ENABLED, self._enabled_to_id(action.get("enabled", "Enabled"))))
                        event_times.append(current_time)
                        events.append(Event(EventType.VFX_DISABLE_OTHERS, self._enabled_to_id(action.get("disableOthers", False))))
                        event_times.append(current_time)
                        props = action.get("filterProperties", "")
                        events.append(Event(EventType.FILTER_PROPERTIES, 0 if not props else 1))
                        event_times.append(current_time)
                    
                    elif event_type == "Bookmark":
                        events.append(Event(EventType.BOOKMARK, 1))
                        event_times.append(current_time)
                    elif event_type == "EditorComment":
                        events.append(Event(EventType.EDITOR_COMMENT, 1))
                        event_times.append(current_time)
                    elif event_type == "CallMethod":
                        events.append(Event(EventType.CALL_METHOD, 1))
                        event_times.append(current_time)
                    elif event_type == "AddComponent":
                        events.append(Event(EventType.ADD_COMPONENT, 1))
                        event_times.append(current_time)
                    elif event_type == "ChangeTrack":
                        events.append(Event(EventType.CHANGE_TRACK, 1))
                        event_times.append(current_time)
                    elif event_type == "FreeRoamWarning":
                        events.append(Event(EventType.FREE_ROAM_WARNING, 1))
                        event_times.append(current_time)
            
            # Handle midspin tiles (999). Token value is 0 (EventRange 0-0); 999 stays on disk.
            if tile_angle == 999:
                events.append(Event(EventType.MIDSPIN, 0))
                event_times.append(current_time)
            else:
                events.append(Event(EventType.TIME_SHIFT, int(current_time)))
                event_times.append(current_time)
                events.append(Event(EventType.TILE_ANGLE, int(round(float(tile_angle))) % 360))
                event_times.append(current_time)

            # Pause is a wait on this tile after arrival, before leaving for the next tile.
            current_time += pending_pause_ms
            pending_pause_ms = 0.0
            
            # Calculate time to next tile using community formula
            # rel = (next_angle - current_heading + 540) % 360
            # if twirl: rel = 360 - rel
            # if rel == 0: rel = 360  (full spin, not instant)
            # ms = (rel / 180) * (60000 / bpm)
            
            if floor_idx < len(level.angle_data) - 1:
                next_angle = float(level.angle_data[floor_idx + 1])
                if next_angle == 999:
                    # Midspin next: outgoing heading = this tile angle, not this+180.
                    # Midspin itself does not consume a travel interval.
                    if tile_angle != 999:
                        current_heading = float(tile_angle)
                    continue

                # Community formula: rel=(next-this+540)%360
                rel = (next_angle - current_heading + 540) % 360
                if not twirl_clockwise:
                    rel = 360 - rel
                if rel == 0:
                    rel = 360

                beats = rel / 180.0
                beats += hold_duration
                hold_duration = 0.0
                # Default gameplay is two planets; ThreePlanets scales interval by 2/3.
                beats *= 2.0 / max(int(multiplanet_count), 1)

                current_time += beats * (60000.0 / current_bpm)
                if tile_angle != 999:
                    current_heading = float(tile_angle)
        
        return events, event_times
    
    def events_to_level(
        self, 
        events: list[Event],
        event_times: list[float],
        base_settings: Optional[dict] = None
    ) -> AdofaiLevel:
        """
        Convert event sequence back to ADOFAI level structure.
        
        Args:
            events: List of osuT5 Event objects
            event_times: List of timestamps in milliseconds
            base_settings: Optional base settings dict to use (for metadata like artist, song, etc.)
            
        Returns:
            AdofaiLevel object
        """
        if base_settings is None:
            base_settings = {}
        
        # Initialize settings with defaults
        settings = {
            "version": 14,
            "artist": base_settings.get("artist", "Unknown Artist"),
            "song": base_settings.get("song", "Unknown Song"),
            "author": base_settings.get("author", "AI Generated"),
            "separateCountdownTime": True,
            "previewSongStart": 0,
            "previewSongDuration": 10,
            "seizureWarning": False,
            "levelDesc": base_settings.get("levelDesc", "Generated by Mapperatorinator ADOFAI"),
            "levelTags": base_settings.get("levelTags", ""),
            "artistPermission": base_settings.get("artistPermission", ""),
            "songFilename": base_settings.get("songFilename", "song.ogg"),
            "bpm": 120,
            "volume": 100,
            "offset": 0,
            "pitch": 100,
            "hitsound": "Kick",
            "hitsoundVolume": 100,
        }
        
        angle_data = []
        actions = []
        
        # Track current floor index
        current_floor = 0
        
        # Parse events
        i = 0
        while i < len(events):
            event = events[i]
            
            if event.type == EventType.BPM:
                settings["bpm"] = event.value
            elif event.type == EventType.DIFFICULTY:
                settings["difficulty"] = event.value
            
            elif event.type == EventType.OFFSET:
                settings["offset"] = event.value

            elif event.type == EventType.PITCH:
                settings["pitch"] = event.value
            
            elif event.type == EventType.TILE_ANGLE:
                angle_data.append(int(event.value))
                current_floor = len(angle_data)
            
            elif event.type == EventType.MIDSPIN:
                angle_data.append(999)
                current_floor = len(angle_data)
            
            elif event.type == EventType.SET_SPEED_BPM:
                speed_action = {
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Bpm",
                    "beatsPerMinute": event.value,
                    "bpmMultiplier": 1.0,
                    "angleOffset": 0,
                }
                if i + 1 < len(events) and events[i + 1].type == EventType.ANGLE_OFFSET:
                    speed_action["angleOffset"] = events[i + 1].value
                    i += 1
                actions.append(speed_action)
            
            elif event.type == EventType.SET_SPEED_MULT:
                speed_action = {
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Multiplier",
                    "beatsPerMinute": settings["bpm"],
                    "bpmMultiplier": event.value / 10.0,
                    "angleOffset": 0,
                }
                if i + 1 < len(events) and events[i + 1].type == EventType.ANGLE_OFFSET:
                    speed_action["angleOffset"] = events[i + 1].value
                    i += 1
                actions.append(speed_action)
            
            elif event.type == EventType.TWIRL:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Twirl"
                })
            
            elif event.type == EventType.PAUSE:
                pause_action = {
                    "floor": current_floor,
                    "eventType": "Pause",
                    "duration": event.value / 10.0,
                    "countdownTicks": 0,
                    "angleCorrectionDir": -1,
                }
                j = i + 1
                while j < len(events) and events[j].type in (EventType.PAUSE_COUNTDOWN, EventType.PAUSE_ANGLE_DIR):
                    if events[j].type == EventType.PAUSE_COUNTDOWN:
                        pause_action["countdownTicks"] = events[j].value
                    else:
                        pause_action["angleCorrectionDir"] = {0: -1, 1: 0, 2: 1}.get(int(events[j].value), -1)
                    j += 1
                actions.append(pause_action)
                i = j - 1
            
            elif event.type == EventType.HOLD:
                hold_action = {
                    "floor": current_floor,
                    "eventType": "Hold",
                    "duration": event.value / 10.0,
                    "distanceMultiplier": 100,
                    "landingAnimation": False,
                }
                j = i + 1
                while j < len(events) and events[j].type in (EventType.HOLD_DISTANCE, EventType.HOLD_LANDING):
                    if events[j].type == EventType.HOLD_DISTANCE:
                        hold_action["distanceMultiplier"] = events[j].value
                    else:
                        hold_action["landingAnimation"] = bool(events[j].value)
                    j += 1
                actions.append(hold_action)
                i = j - 1
            
            elif event.type == EventType.MULTI_PLANET:
                planets_val = int(event.value)
                planets_str = "TwoPlanets" if planets_val == 2 else "ThreePlanets"
                actions.append({
                    "floor": current_floor,
                    "eventType": "MultiPlanet",
                    "planets": planets_str
                })
            
            # Camera events
            elif event.type == EventType.MOVE_CAMERA:
                # MoveCamera followed by position/rotation/zoom/duration/ease/relative
                camera_action = {
                    "floor": current_floor,
                    "eventType": "MoveCamera",
                    "position": [0, 0],
                    "rotation": 0,
                    "zoom": 100,
                    "duration": 1.0,
                    "ease": "Linear",
                    "relativeTo": "Player",
                    "angleOffset": 0,
                    "eventTag": ""
                }
                # Parse following parameter events
                j = i + 1
                while j < len(events) and events[j].type in (
                    EventType.CAMERA_POSITION_X, EventType.CAMERA_POSITION_Y,
                    EventType.CAMERA_ROTATION, EventType.CAMERA_ZOOM,
                    EventType.CAMERA_DURATION, EventType.CAMERA_EASE,
                    EventType.CAMERA_RELATIVE, EventType.ANGLE_OFFSET,
                ):
                    if events[j].type == EventType.CAMERA_POSITION_X:
                        camera_action["position"][0] = events[j].value
                    elif events[j].type == EventType.CAMERA_POSITION_Y:
                        camera_action["position"][1] = events[j].value
                    elif events[j].type == EventType.CAMERA_ROTATION:
                        camera_action["rotation"] = events[j].value
                    elif events[j].type == EventType.CAMERA_ZOOM:
                        camera_action["zoom"] = events[j].value
                    elif events[j].type == EventType.CAMERA_DURATION:
                        camera_action["duration"] = events[j].value / 10.0
                    elif events[j].type == EventType.CAMERA_EASE:
                        camera_action["ease"] = self._id_to_ease(int(events[j].value))
                    elif events[j].type == EventType.CAMERA_RELATIVE:
                        camera_action["relativeTo"] = self._id_to_relative(int(events[j].value))
                    elif events[j].type == EventType.ANGLE_OFFSET:
                        camera_action["angleOffset"] = events[j].value
                    j += 1
                actions.append(camera_action)
                i = j - 1  # Skip parameter events
            
            # Must-have gameplay events
            elif event.type == EventType.CHECKPOINT:
                actions.append({"floor": current_floor, "eventType": "Checkpoint"})
            elif event.type == EventType.AUTO_PLAY_TILES:
                enabled = "Enabled" if event.value else "Disabled"
                actions.append({"floor": current_floor, "eventType": "AutoPlayTiles", "enabled": enabled})
            elif event.type == EventType.SET_PLANET_ROTATION:
                actions.append({"floor": current_floor, "eventType": "SetPlanetRotation", "easeParts": int(event.value)})
            elif event.type == EventType.FREE_ROAM:
                actions.append({"floor": current_floor, "eventType": "FreeRoam"})
            elif event.type == EventType.FREE_ROAM_TWIRL:
                actions.append({"floor": current_floor, "eventType": "FreeRoamTwirl"})
            elif event.type == EventType.FREE_ROAM_REMOVE:
                actions.append({"floor": current_floor, "eventType": "FreeRoamRemove"})
            elif event.type == EventType.SCALE_MARGIN:
                actions.append({"floor": current_floor, "eventType": "ScaleMargin", "scale": int(event.value)})
            elif event.type == EventType.SCALE_RADIUS:
                actions.append({"floor": current_floor, "eventType": "ScaleRadius", "scale": int(event.value)})
            elif event.type == EventType.MULTITAP:
                actions.append({"floor": current_floor, "eventType": "Multitap", "presses": int(event.value)})
            elif event.type == EventType.HIDE:
                actions.append({"floor": current_floor, "eventType": "Hide"})
            elif event.type == EventType.KILL_PLAYER:
                actions.append({"floor": current_floor, "eventType": "KillPlayer"})
            
            # Track events
            elif event.type == EventType.POSITION_TRACK:
                actions.append({"floor": current_floor, "eventType": "PositionTrack", "position": [0, 0]})
            elif event.type == EventType.MOVE_TRACK:
                track_action = {
                    "floor": current_floor,
                    "eventType": "MoveTrack",
                    "positionOffset": [0, 0],
                    "startTile": [0, "ThisTile"],
                    "endTile": [0, "ThisTile"],
                    "duration": 1.0,
                    "ease": "Linear",
                    "angleOffset": 0,
                    "eventTag": "",
                }
                j = i + 1
                while j < len(events) and events[j].type in (
                    EventType.CAMERA_POSITION_X, EventType.CAMERA_POSITION_Y,
                    EventType.TRACK_START_TILE, EventType.TRACK_END_TILE,
                    EventType.CAMERA_DURATION, EventType.CAMERA_EASE, EventType.ANGLE_OFFSET,
                ):
                    if events[j].type == EventType.CAMERA_POSITION_X:
                        track_action["positionOffset"][0] = events[j].value
                    elif events[j].type == EventType.CAMERA_POSITION_Y:
                        track_action["positionOffset"][1] = events[j].value
                    elif events[j].type == EventType.TRACK_START_TILE:
                        track_action["startTile"] = self._id_to_tile_ref(events[j].value)
                    elif events[j].type == EventType.TRACK_END_TILE:
                        track_action["endTile"] = self._id_to_tile_ref(events[j].value)
                    elif events[j].type == EventType.CAMERA_DURATION:
                        track_action["duration"] = events[j].value / 10.0
                    elif events[j].type == EventType.CAMERA_EASE:
                        track_action["ease"] = self._id_to_ease(int(events[j].value))
                    elif events[j].type == EventType.ANGLE_OFFSET:
                        track_action["angleOffset"] = events[j].value
                    j += 1
                actions.append(track_action)
                i = j - 1
            elif event.type == EventType.COLOR_TRACK:
                actions.append({"floor": current_floor, "eventType": "ColorTrack"})
            elif event.type == EventType.ANIMATE_TRACK:
                actions.append({"floor": current_floor, "eventType": "AnimateTrack"})
            
            # Audio events
            elif event.type == EventType.SET_HITSOUND:
                actions.append({"floor": current_floor, "eventType": "SetHitsound"})
            elif event.type == EventType.PLAY_SOUND:
                actions.append({"floor": current_floor, "eventType": "PlaySound"})
            elif event.type == EventType.SET_HOLD_SOUND:
                actions.append({"floor": current_floor, "eventType": "SetHoldSound"})
            
            # Event control
            elif event.type == EventType.REPEAT_EVENTS:
                actions.append({"floor": current_floor, "eventType": "RepeatEvents"})
            elif event.type == EventType.SET_CONDITIONAL_EVENTS:
                actions.append({"floor": current_floor, "eventType": "SetConditionalEvents"})
            elif event.type == EventType.SET_INPUT_EVENT:
                actions.append({"floor": current_floor, "eventType": "SetInputEvent"})
            
            elif event.type == EventType.FLASH:
                flash = {
                    "floor": current_floor,
                    "eventType": "Flash",
                    "duration": event.value / 10.0,
                    "plane": "Foreground",
                    "startColor": "ffffff",
                    "startOpacity": 100,
                    "endColor": "ffffff",
                    "endOpacity": 0,
                    "angleOffset": 0,
                    "ease": "Linear",
                    "eventTag": "",
                }
                j = i + 1
                color_step = 0
                opacity_step = 0
                while j < len(events) and events[j].type in (
                    EventType.VFX_PLANE, EventType.VFX_COLOR, EventType.VFX_OPACITY,
                    EventType.ANGLE_OFFSET, EventType.CAMERA_EASE,
                ):
                    if events[j].type == EventType.VFX_PLANE:
                        flash["plane"] = "Background" if events[j].value == 0 else "Foreground"
                    elif events[j].type == EventType.VFX_COLOR:
                        hex_color = self._color_id_to_hex(events[j].value)
                        if color_step == 0:
                            flash["startColor"] = hex_color
                        else:
                            flash["endColor"] = hex_color
                        color_step += 1
                    elif events[j].type == EventType.VFX_OPACITY:
                        if opacity_step == 0:
                            flash["startOpacity"] = events[j].value
                        else:
                            flash["endOpacity"] = events[j].value
                        opacity_step += 1
                    elif events[j].type == EventType.ANGLE_OFFSET:
                        flash["angleOffset"] = events[j].value
                    elif events[j].type == EventType.CAMERA_EASE:
                        flash["ease"] = self._id_to_ease(int(events[j].value))
                    j += 1
                actions.append(flash)
                i = j - 1
            elif event.type == EventType.BLOOM:
                bloom = {
                    "floor": current_floor,
                    "eventType": "Bloom",
                    "enabled": "Enabled" if int(event.value) > 0 else "Disabled",
                    "threshold": 50,
                    "intensity": int(event.value),
                    "color": "ffffff",
                }
                j = i + 1
                while j < len(events) and events[j].type in (
                    EventType.VFX_ENABLED, EventType.VFX_COLOR, EventType.VFX_THRESHOLD,
                ):
                    if events[j].type == EventType.VFX_ENABLED:
                        bloom["enabled"] = self._id_to_enabled(events[j].value)
                    elif events[j].type == EventType.VFX_THRESHOLD:
                        bloom["threshold"] = events[j].value
                    else:
                        bloom["color"] = self._color_id_to_hex(events[j].value)
                    j += 1
                actions.append(bloom)
                i = j - 1
            elif event.type == EventType.SHAKE_SCREEN:
                shake = {
                    "floor": current_floor,
                    "eventType": "ShakeScreen",
                    "duration": 1.0,
                    "strength": 100,
                    "intensity": int(event.value),
                    "fadeOut": "Enabled",
                }
                if i + 1 < len(events) and events[i + 1].type == EventType.VFX_STRENGTH:
                    shake["strength"] = events[i + 1].value
                    i += 1
                actions.append(shake)
            elif event.type == EventType.SET_FILTER:
                filt = {
                    "floor": current_floor,
                    "eventType": "SetFilter",
                    "filter": self._id_to_filter(int(event.value)),
                    "enabled": "Enabled",
                    "intensity": 100,
                    "disableOthers": "Disabled",
                }
                j = i + 1
                while j < len(events) and events[j].type in (
                    EventType.VFX_ENABLED, EventType.VFX_INTENSITY, EventType.VFX_DISABLE_OTHERS,
                ):
                    if events[j].type == EventType.VFX_ENABLED:
                        filt["enabled"] = self._id_to_enabled(events[j].value)
                    elif events[j].type == EventType.VFX_INTENSITY:
                        filt["intensity"] = events[j].value
                    else:
                        filt["disableOthers"] = self._id_to_enabled(events[j].value)
                    j += 1
                actions.append(filt)
                i = j - 1
            elif event.type == EventType.SET_FILTER_ADVANCED:
                adv = {
                    "floor": current_floor,
                    "eventType": "SetFilterAdvanced",
                    "filter": self._id_to_filter(int(event.value)),
                    "enabled": "Enabled",
                    "disableOthers": "Disabled",
                    "filterProperties": "",
                }
                j = i + 1
                while j < len(events) and events[j].type in (
                    EventType.VFX_ENABLED, EventType.VFX_DISABLE_OTHERS, EventType.FILTER_PROPERTIES,
                ):
                    if events[j].type == EventType.VFX_ENABLED:
                        adv["enabled"] = self._id_to_enabled(events[j].value)
                    elif events[j].type == EventType.VFX_DISABLE_OTHERS:
                        adv["disableOthers"] = self._id_to_enabled(events[j].value)
                    else:
                        adv["filterProperties"] = "" if events[j].value == 0 else "1"
                    j += 1
                actions.append(adv)
                i = j - 1
            elif event.type == EventType.BOOKMARK:
                actions.append({"floor": current_floor, "eventType": "Bookmark"})
            elif event.type == EventType.EDITOR_COMMENT:
                actions.append({"floor": current_floor, "eventType": "EditorComment"})
            elif event.type == EventType.CALL_METHOD:
                actions.append({"floor": current_floor, "eventType": "CallMethod"})
            elif event.type == EventType.ADD_COMPONENT:
                actions.append({"floor": current_floor, "eventType": "AddComponent"})
            elif event.type == EventType.CHANGE_TRACK:
                actions.append({"floor": current_floor, "eventType": "ChangeTrack"})
            elif event.type == EventType.FREE_ROAM_WARNING:
                actions.append({"floor": current_floor, "eventType": "FreeRoamWarning"})
            
            i += 1
        
        # If no angles were found, add a minimal valid path
        if not angle_data:
            angle_data = [0, 0, 0]  # Minimal 3-tile straight path
        
        return AdofaiLevel(
            settings=settings,
            angle_data=angle_data,
            actions=actions,
            decorations=[]
        )
