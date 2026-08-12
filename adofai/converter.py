"""
Converter between ADOFAI events and model tokens.

This module bridges ADOFAI's event representation and the tokenizer vocabulary,
handling conversion to/from the intermediate event format used by the model.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .event import AdofaiEvent, AdofaiEventType, AdofaiEventRange
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
    
    def level_to_events(self, level: AdofaiLevel) -> tuple[list[AdofaiEvent], list[float]]:
        """
        Convert ADOFAI level to time-ordered event sequence.
        
        Args:
            level: Parsed ADOFAI level
            
        Returns:
            Tuple of (events, event_times) where:
                - events: List of AdofaiEvent objects
                - event_times: List of timestamps in milliseconds
        """
        events = []
        event_times = []
        
        # Extract initial settings
        settings = level.settings
        initial_bpm = settings.get("bpm", 100)
        offset = settings.get("offset", 0)
        
        # Add metadata events at the start
        events.append(AdofaiEvent(AdofaiEventType.BPM, initial_bpm))
        event_times.append(0)
        
        events.append(AdofaiEvent(AdofaiEventType.OFFSET, offset))
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
                            events.append(AdofaiEvent(AdofaiEventType.SET_SPEED_BPM, new_bpm))
                            event_times.append(current_time)
                            current_bpm = new_bpm
                        elif speed_type == "Multiplier":
                            multiplier = action.get("bpmMultiplier", 1.0)
                            events.append(AdofaiEvent(AdofaiEventType.SET_SPEED_MULT, multiplier))
                            event_times.append(current_time)
                            current_bpm *= multiplier
                    
                    elif event_type == "Twirl":
                        twirl_clockwise = not twirl_clockwise
                        events.append(AdofaiEvent(AdofaiEventType.TWIRL, 1))
                        event_times.append(current_time)
                    
                    elif event_type == "Pause":
                        duration = action.get("duration", 1.0)
                        events.append(AdofaiEvent(AdofaiEventType.PAUSE, duration))
                        event_times.append(current_time)
                    
                    elif event_type == "Hold":
                        duration = action.get("duration", 1.0)
                        events.append(AdofaiEvent(AdofaiEventType.HOLD, duration))
                        event_times.append(current_time)
                    
                    elif event_type == "MultiPlanet":
                        planets = action.get("planets", 2)
                        events.append(AdofaiEvent(AdofaiEventType.MULTI_PLANET, planets))
                        event_times.append(current_time)
            
            # Handle midspin tiles (999)
            if tile_angle == 999:
                events.append(AdofaiEvent(AdofaiEventType.MIDSPIN, 999))
                event_times.append(current_time)
                # Midspin doesn't advance time, planet continues from previous angle
                continue
            
            # Add tile angle event
            events.append(AdofaiEvent(AdofaiEventType.TIME_SHIFT, int(current_time)))
            event_times.append(current_time)
            
            events.append(AdofaiEvent(AdofaiEventType.TILE_ANGLE, tile_angle))
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
        events: list[AdofaiEvent],
        event_times: list[float],
        base_settings: Optional[dict] = None
    ) -> AdofaiLevel:
        """
        Convert event sequence back to ADOFAI level structure.
        
        Args:
            events: List of AdofaiEvent objects
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
            
            if event.type == AdofaiEventType.BPM:
                settings["bpm"] = event.value
            
            elif event.type == AdofaiEventType.OFFSET:
                settings["offset"] = event.value
            
            elif event.type == AdofaiEventType.TILE_ANGLE:
                angle_data.append(int(event.value))
                current_floor = len(angle_data)
            
            elif event.type == AdofaiEventType.MIDSPIN:
                angle_data.append(999)
                current_floor = len(angle_data)
            
            elif event.type == AdofaiEventType.SET_SPEED_BPM:
                actions.append({
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Bpm",
                    "beatsPerMinute": event.value,
                    "bpmMultiplier": 1.0
                })
            
            elif event.type == AdofaiEventType.SET_SPEED_MULT:
                actions.append({
                    "floor": current_floor,
                    "eventType": "SetSpeed",
                    "speedType": "Multiplier",
                    "beatsPerMinute": settings["bpm"],
                    "bpmMultiplier": event.value
                })
            
            elif event.type == AdofaiEventType.TWIRL:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Twirl"
                })
            
            elif event.type == AdofaiEventType.PAUSE:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Pause",
                    "duration": event.value,
                    "countdownTicks": 0,
                    "angleCorrectionDir": -1
                })
            
            elif event.type == AdofaiEventType.HOLD:
                actions.append({
                    "floor": current_floor,
                    "eventType": "Hold",
                    "duration": event.value,
                    "distanceMultiplier": 100
                })
            
            elif event.type == AdofaiEventType.MULTI_PLANET:
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
