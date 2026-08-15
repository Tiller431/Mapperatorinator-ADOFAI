"""
Converter between ADOFAI events and model tokens.

This module bridges ADOFAI's event representation and the tokenizer vocabulary,
handling conversion to/from the intermediate event format used by the model.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

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
        
        # Add difficulty if available
        difficulty = settings.get("difficulty", None)
        if difficulty is not None:
            events.append(Event(EventType.DIFFICULTY, int(difficulty)))
            event_times.append(0)
        
        # Track state for timing calculations
        current_time = offset
        current_bpm = initial_bpm
        current_angle = 0.0  # Track cumulative angle for rotation calculations
        twirl_clockwise = True  # Default rotation direction
        
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
                            events.append(Event(EventType.SET_SPEED_BPM, new_bpm))
                            event_times.append(current_time)
                            current_bpm = new_bpm
                        elif speed_type == "Multiplier":
                            multiplier = action.get("bpmMultiplier", 1.0)
                            events.append(Event(EventType.SET_SPEED_MULT, multiplier))
                            event_times.append(current_time)
                            current_bpm *= multiplier
                    
                    elif event_type == "Twirl":
                        twirl_clockwise = not twirl_clockwise
                        events.append(Event(EventType.TWIRL, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "Pause":
                        duration = action.get("duration", 1.0)
                        events.append(Event(EventType.PAUSE, duration))
                        event_times.append(current_time)
                    
                    elif event_type == "Hold":
                        duration = action.get("duration", 1.0)
                        events.append(Event(EventType.HOLD, duration))
                        event_times.append(current_time)
                    
                    elif event_type == "MultiPlanet":
                        planets = action.get("planets", 2)
                        events.append(Event(EventType.MULTI_PLANET, planets))
                        event_times.append(current_time)
                    
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
                        
                        # Duration in beats
                        duration = action.get("duration", 1.0)
                        events.append(Event(EventType.CAMERA_DURATION, duration))
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
                    
                    elif event_type == "MoveTrack":
                        events.append(Event(EventType.MOVE_TRACK, 1))
                        event_times.append(current_time)
                        pos = action.get("position", [0, 0])
                        events.append(Event(EventType.CAMERA_POSITION_X, int(pos[0])))
                        event_times.append(current_time)
                        events.append(Event(EventType.CAMERA_POSITION_Y, int(pos[1])))
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
                    
                    # VFX events (Tyler override) - SharpFAI/MagicShaper field names
                    elif event_type == "Flash":
                        # SharpFAI fields: duration, plane, startColor, startOpacity, endColor, endOpacity, angleOffset, ease, eventTag
                        duration = action.get("duration", 1.0)
                        events.append(Event(EventType.FLASH, int(duration * 10)))  # Quantize to 0.1 beat steps
                        event_times.append(current_time)
                    
                    elif event_type == "Bloom":
                        # SharpFAI fields: enabled, threshold, intensity, color, duration, ease, angleOffset, eventTag
                        enabled = action.get("enabled", "Enabled")
                        if isinstance(enabled, str):
                            enabled = 1 if enabled == "Enabled" else 0
                        intensity = action.get("intensity", 100)
                        events.append(Event(EventType.BLOOM, int(intensity) if enabled else 0))
                        event_times.append(current_time)
                    
                    elif event_type == "ShakeScreen":
                        # SharpFAI fields: duration, strength, intensity, ease, fadeOut, angleOffset, eventTag
                        # NOTE: Gitbook "Speed" field is actually "intensity" on-disk, NOT "speed"
                        intensity = action.get("intensity", 100)
                        events.append(Event(EventType.SHAKE_SCREEN, int(intensity)))
                        event_times.append(current_time)
                    
                    elif event_type == "SetFilter":
                        # SharpFAI fields: filter, enabled, intensity, duration, ease, disableOthers, angleOffset, eventTag
                        # MagicShaper filter enum (NOT ADOFAI-JS wrong names like Bloom/Pixellate/etc.)
                        filter_name = action.get("filter", "None")
                        # Map MagicShaper filters to IDs (subset of 40+ filters)
                        filter_map = {
                            "None": 0, "Grayscale": 1, "Sepia": 2, "Invert": 3, "VHS": 4,
                            "EightiesTV": 5, "FiftiesTV": 6, "Arcade": 7, "LED": 8, "Rain": 9,
                            "Blizzard": 10, "PixelSnow": 11, "Compression": 12, "Glitch": 13,
                            "Pixelate": 14, "Waves": 15, "Static": 16, "Grain": 17, "MotionBlur": 18,
                            "Fisheye": 19, "Aberration": 20, "Drawing": 21, "Neon": 22, "Handheld": 23,
                            "NightVision": 24, "Funk": 25, "Tunnel": 26, "Weird3D": 27, "Blur": 28,
                            "GaussianBlur": 29, "Posterize": 30
                        }
                        filter_id = filter_map.get(filter_name, 0)
                        events.append(Event(EventType.SET_FILTER, filter_id))
                        event_times.append(current_time)
            
            # Handle midspin tiles (999)
            if tile_angle == 999:
                events.append(Event(EventType.MIDSPIN, 999))
                event_times.append(current_time)
                # Midspin doesn't advance time, planet continues from previous angle
                continue
            
            # Add tile angle event
            events.append(Event(EventType.TIME_SHIFT, int(current_time)))
            event_times.append(current_time)
            
            events.append(Event(EventType.TILE_ANGLE, tile_angle))
            event_times.append(current_time)
            
            # Calculate time to next tile based on angle rotation
            if floor_idx < len(level.angle_data) - 1:
                # Calculate angle difference
                target_angle = float(tile_angle)
                if twirl_clockwise:
                    angle_diff = current_angle - target_angle
                else:
                    angle_diff = target_angle - current_angle
                
                # Normalize to positive angle
                while angle_diff < 0:
                    angle_diff += 360
                while angle_diff >= 360:
                    angle_diff -= 360
                
                # Convert angle to time: 180 degrees = 1 beat
                beats = angle_diff / 180.0
                ms_per_beat = 60000.0 / current_bpm
                time_delta = beats * ms_per_beat
                
                current_time += time_delta
                
                # Update current angle for next iteration
                # After hitting this tile, planet is at opposite side
                current_angle = (target_angle + 180.0) % 360.0
        
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
            
            elif event.type == EventType.OFFSET:
                settings["offset"] = event.value
            
            elif event.type == EventType.TILE_ANGLE:
                angle_data.append(int(event.value))
                current_floor = len(angle_data)
            
            elif event.type == EventType.MIDSPIN:
                angle_data.append(999)
                current_floor = len(angle_data)
            
            elif event.type == EventType.SET_SPEED_BPM:
                actions.append({
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Bpm",
                    "beatsPerMinute": event.value,
                    "bpmMultiplier": 1.0
                })
            
            elif event.type == EventType.SET_SPEED_MULT:
                actions.append({
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Multiplier",
                    "beatsPerMinute": settings["bpm"],
                    "bpmMultiplier": event.value
                })
            
            elif event.type == EventType.TWIRL:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Twirl"
                })
            
            elif event.type == EventType.PAUSE:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Pause",
                    "duration": event.value,
                    "countdownTicks": 0,
                    "angleCorrectionDir": -1
                })
            
            elif event.type == EventType.HOLD:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Hold",
                    "duration": event.value,
                    "distanceMultiplier": 100
                })
            
            elif event.type == EventType.MULTI_PLANET:
                actions.append({
                    "floor": current_floor,
                    "eventType": "MultiPlanet",
                    "planets": int(event.value)
                })
            
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
