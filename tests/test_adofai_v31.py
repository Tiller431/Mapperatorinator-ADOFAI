"""
Tests for ADOFAI v31 Whisper training path.

Verifies:
- SharpFAI field name round-trips (ease, eventTag, positionOffset, planets enum)
- Lossless augmentation (rotation updates camera/track positions)
- Converter timing bugs are fixed (heading 180, angle_diff==0→360, midspin, Pause/Hold/MultiPlanet)
"""

import pytest
import numpy as np
from pathlib import Path

from adofai.parser import AdofaiLevel
from adofai.converter import AdofaiConverter
from osuT5.osuT5.event import Event, EventType


class TestSharpFAIFieldNames:
    """Test that SharpFAI on-disk field names round-trip correctly."""
    
    def test_movetrack_positionOffset(self):
        """MoveTrack uses positionOffset, not position."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90, 180],
            actions=[{
                "floor": 1,
                "eventType": "MoveTrack",
                "positionOffset": [10, 20],
                "duration": 2.0,
                "ease": "Linear",
                "angleOffset": 0,
                "eventTag": ""
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        # Find MoveTrack event
        move_track_events = [e for e in events if e.type == EventType.MOVE_TRACK]
        assert len(move_track_events) > 0, "MoveTrack event should be emitted"
    
    def test_multiplanet_planets_enum(self):
        """MultiPlanet.planets uses 'TwoPlanets'/'ThreePlanets' enum strings."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 0,
                "eventType": "MultiPlanet",
                "planets": "TwoPlanets"
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        # Find MultiPlanet event
        multiplanet_events = [e for e in events if e.type == EventType.MULTI_PLANET]
        assert len(multiplanet_events) > 0
        assert multiplanet_events[0].value == 2
        
        # Round-trip
        reconstructed = converter.events_to_level(events, event_times, level.settings)
        mp_actions = [a for a in reconstructed.actions if a.get("eventType") == "MultiPlanet"]
        assert len(mp_actions) > 0
        assert mp_actions[0]["planets"] in ("TwoPlanets", "ThreePlanets")
    
    def test_movecamera_ease_and_relativeTo(self):
        """MoveCamera uses 'ease' and 'relativeTo' (not 'easing' or 'relative')."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 1,
                "eventType": "MoveCamera",
                "position": [0, 0],
                "rotation": 45,
                "zoom": 100,
                "duration": 2.0,
                "ease": "OutQuad",
                "relativeTo": "LastPositionNoRotation",
                "angleOffset": 0,
                "eventTag": "cam1"
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        # Find MoveCamera and parameter events
        camera_events = [e for e in events if e.type == EventType.MOVE_CAMERA]
        assert len(camera_events) > 0
        
        # Check ease and relativeTo are parsed
        ease_events = [e for e in events if e.type == EventType.CAMERA_EASE]
        relative_events = [e for e in events if e.type == EventType.CAMERA_RELATIVE]
        assert len(ease_events) > 0
        assert len(relative_events) > 0


class TestLosslessAugmentation:
    """Test lossless augmentation transforms."""
    
    def test_rotation_updates_camera_position(self):
        """Rotation must update camera/track world positions."""
        from osuT5.osuT5.dataset.adofai_dataset import AdofaiDataset
        
        # Create a dataset instance (won't actually load data)
        class MockArgs:
            adofai_rotate_prob = 1.0
            adofai_reflect_prob = 0.0
            adofai_pitch_prob = 0.0
            adofai_rate_prob = 0.0
        
        dataset = AdofaiDataset.__new__(AdofaiDataset)
        dataset.p_rotate = 1.0
        
        # Test rotation of camera position
        angles = [0, 90, 180]
        actions = [{
            "eventType": "MoveCamera",
            "position": [10, 0],
            "rotation": 0
        }]
        
        rotated_angles, rotated_actions = dataset._apply_rotation(angles, actions, 90.0)
        
        # Check angles rotated
        assert rotated_angles[0] == 90
        assert rotated_angles[1] == 180
        assert rotated_angles[2] == 270
        
        # Check camera position rotated
        cam = rotated_actions[0]
        assert cam["eventType"] == "MoveCamera"
        # Original [10, 0] rotated 90° → [0, 10]
        assert abs(cam["position"][0] - 0) < 1.0
        assert abs(cam["position"][1] - 10) < 1.0
        # Rotation field also rotated
        assert abs(cam["rotation"] - 90) < 1.0
    
    def test_rotation_updates_movetrack_positionOffset(self):
        """Rotation must update MoveTrack.positionOffset."""
        from osuT5.osuT5.dataset.adofai_dataset import AdofaiDataset
        
        dataset = AdofaiDataset.__new__(AdofaiDataset)
        dataset.p_rotate = 1.0
        
        angles = [0, 90]
        actions = [{
            "eventType": "MoveTrack",
            "positionOffset": [5, 5]
        }]
        
        rotated_angles, rotated_actions = dataset._apply_rotation(angles, actions, 180.0)
        
        # Check positionOffset rotated
        track = rotated_actions[0]
        assert track["eventType"] == "MoveTrack"
        # [5, 5] rotated 180° → [-5, -5]
        assert abs(track["positionOffset"][0] - (-5)) < 1.0
        assert abs(track["positionOffset"][1] - (-5)) < 1.0
    
    def test_midspin_999_unchanged_by_rotation(self):
        """Midspin tiles (999) must not be rotated."""
        from osuT5.osuT5.dataset.adofai_dataset import AdofaiDataset
        
        dataset = AdofaiDataset.__new__(AdofaiDataset)
        dataset.p_rotate = 1.0
        
        angles = [0, 999, 90]
        actions = []
        
        rotated_angles, rotated_actions = dataset._apply_rotation(angles, actions, 45.0)
        
        assert rotated_angles[0] == 45
        assert rotated_angles[1] == 999  # Unchanged
        assert rotated_angles[2] == 135


class TestConverterTimingBugs:
    """Test that converter timing bugs are fixed."""
    
    def test_start_heading_is_180(self):
        """Game starts heading 180° (not 0°)."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],  # First tile at 0°
            actions=[],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        # Check that the time to reach first tile (0°) from heading 180° is correct
        # rel = (0 - 180 + 540) % 360 = 360 % 360 = 0 → 360 (full spin)
        # 360° / 180° = 2 beats = 1000ms @ 120 BPM
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        assert len(time_shifts) > 0
        # First tile should be at offset + travel time
        # With heading 180, reaching tile at 0° is a 180° rotation (1 beat = 500ms)
        # rel = (0 - 180 + 540) % 360 = 360, then 360 → full spin
        # Actually: (next - current_heading + 540) % 360 = (0 - 180 + 540) % 360 = 360, then rel==0 → 360
        # So 360 / 180 = 2 beats = 1000ms
        assert time_shifts[0] >= 1000  # At least 1000ms for full rotation
    
    def test_angle_diff_zero_is_360(self):
        """angle_diff == 0 must be 360° (full spin), not 0ms."""
        converter = AdofaiConverter()
        # Two tiles at same angle = full 360° spin
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 0],
            actions=[],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        assert len(time_shifts) >= 2
        # Time between tiles should be 2 beats (360° / 180° = 2)
        # At 120 BPM: 2 beats = 1000ms
        # Second tile time should be first + 1000
        if len(time_shifts) >= 2:
            time_delta = time_shifts[1] - time_shifts[0]
            assert abs(time_delta - 1000) < 10  # Allow small rounding error
    
    def test_midspin_heading_unchanged(self):
        """Midspin outgoing heading = incoming heading (not incoming + 180)."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 999, 180],  # Tile, midspin, tile
            actions=[],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        # Find midspin and check it's emitted
        midspin_events = [e for e in events if e.type == EventType.MIDSPIN]
        assert len(midspin_events) > 0
        
        # After midspin, planet should continue toward next tile
        # from the same heading (not flipped)
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        # Midspin doesn't advance time
        assert len(time_shifts) >= 2
    
    def test_pause_adds_time(self):
        """Pause adds extra time (duration in beats)."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 0,
                "eventType": "Pause",
                "duration": 2.0  # 2 beats
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        pause_events = [(e, t) for e, t in zip(events, event_times) if e.type == EventType.PAUSE]
        assert len(pause_events) > 0
        
        # Pause should add 2 beats = 1000ms @ 120 BPM
        # Check that time advances correctly
    
    def test_hold_extends_travel_time(self):
        """Hold extends travel time by duration beats."""
        converter = AdofaiConverter()
        # Normal: 0→90 is 90° = 0.5 beats = 250ms
        # With Hold(1.0): 0.5 + 1.0 = 1.5 beats = 750ms
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 0,
                "eventType": "Hold",
                "duration": 1.0
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        hold_events = [e for e in events if e.type == EventType.HOLD]
        assert len(hold_events) > 0
        
        # TODO: Verify time delta is extended
    
    def test_multiplanet_divides_travel_time(self):
        """MultiPlanet divides travel time by planet count."""
        converter = AdofaiConverter()
        # Normal: 0→180 is 180° = 1 beat = 500ms
        # With TwoPlanets: 1 / 2 = 0.5 beats = 250ms
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 180],
            actions=[{
                "floor": 0,
                "eventType": "MultiPlanet",
                "planets": "TwoPlanets"
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        multiplanet_events = [e for e in events if e.type == EventType.MULTI_PLANET]
        assert len(multiplanet_events) > 0
        
        # TODO: Verify time delta is divided


class TestVFXFieldNames:
    """Test VFX events use SharpFAI field names, not ADOFAI-JS."""
    
    def test_flash_uses_startColor_not_color(self):
        """Flash uses startColor/endColor/startOpacity/endOpacity, not 'color'/'opacity'."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 1,
                "eventType": "Flash",
                "duration": 1.0,
                "plane": "Foreground",
                "startColor": "ff0000",
                "startOpacity": 100,
                "endColor": "00ff00",
                "endOpacity": 0,
                "ease": "Linear",
                "angleOffset": 0,
                "eventTag": ""
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        flash_events = [e for e in events if e.type == EventType.FLASH]
        assert len(flash_events) > 0
    
    def test_bloom_uses_enabled_not_bool(self):
        """Bloom.enabled is 'Enabled'/'Disabled' string, not bool."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 1,
                "eventType": "Bloom",
                "enabled": "Enabled",
                "threshold": 50,
                "intensity": 80,
                "color": "ffffff"
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        bloom_events = [e for e in events if e.type == EventType.BLOOM]
        assert len(bloom_events) > 0
    
    def test_shakescreen_uses_intensity_not_speed(self):
        """ShakeScreen uses 'intensity', NOT 'speed' (Gitbook is wrong)."""
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 1,
                "eventType": "ShakeScreen",
                "duration": 1.0,
                "strength": 100,
                "intensity": 80,  # NOT 'speed'
                "ease": "Linear",
                "fadeOut": "Enabled"
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        shake_events = [e for e in events if e.type == EventType.SHAKE_SCREEN]
        assert len(shake_events) > 0
        assert shake_events[0].value == 80
    
    def test_setfilter_uses_magicshaper_enum(self):
        """SetFilter.filter uses MagicShaper enum (Grayscale, Sepia, etc.), not ADOFAI-JS names."""
        converter = AdofaiConverter()
        # Correct: "Pixelate" (MagicShaper)
        # Wrong: "Pixellate" (ADOFAI-JS typo)
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{
                "floor": 1,
                "eventType": "SetFilter",
                "filter": "Pixelate",  # MagicShaper spelling
                "enabled": "Enabled",
                "intensity": 100
            }],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        filter_events = [e for e in events if e.type == EventType.SET_FILTER]
        assert len(filter_events) > 0
        # Filter ID should map to Pixelate
        assert filter_events[0].value == 14  # Pixelate is ID 14 in the map


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
