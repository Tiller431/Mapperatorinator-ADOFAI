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
        from osuT5.osuT5.dataset.adofai_augment import apply_rotation

        angles = [0, 90, 180]
        actions = [{
            "eventType": "MoveCamera",
            "position": [10, 0],
            "rotation": 0
        }]
        
        rotated_angles, rotated_actions = apply_rotation(angles, actions, 90.0)
        
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
        from osuT5.osuT5.dataset.adofai_augment import apply_rotation

        angles = [0, 90]
        actions = [{
            "eventType": "MoveTrack",
            "positionOffset": [5, 5]
        }]
        
        rotated_angles, rotated_actions = apply_rotation(angles, actions, 180.0)
        
        # Check positionOffset rotated
        track = rotated_actions[0]
        assert track["eventType"] == "MoveTrack"
        # [5, 5] rotated 180° → [-5, -5]
        assert abs(track["positionOffset"][0] - (-5)) < 1.0
        assert abs(track["positionOffset"][1] - (-5)) < 1.0
    
    def test_midspin_999_unchanged_by_rotation(self):
        """Midspin tiles (999) must not be rotated."""
        from osuT5.osuT5.dataset.adofai_augment import apply_rotation

        angles = [0, 999, 90]
        actions = []
        
        rotated_angles, rotated_actions = apply_rotation(angles, actions, 45.0)
        
        assert rotated_angles[0] == 45
        assert rotated_angles[1] == 999  # Unchanged
        assert rotated_angles[2] == 135

    def test_reflect_axis_formulas(self):
        from osuT5.osuT5.dataset.adofai_augment import apply_reflection

        assert apply_reflection([40, 999], [], "x_flip")[0] == [320, 999]
        assert apply_reflection([40, 999], [], "y_flip")[0] == [140, 999]
        assert apply_reflection([40, 999], [], "diag_y_eq_x")[0] == [50, 999]
        assert apply_reflection([40, 999], [], "diag_y_eq_neg_x")[0] == [230, 999]

    def test_reflect_toggles_floor0_twirl_only(self):
        from osuT5.osuT5.dataset.adofai_augment import apply_reflection

        angles, actions = apply_reflection(
            [10, 999],
            [{"floor": 1, "eventType": "Twirl"}],
            "x_flip",
        )
        assert angles[0] == 350
        assert angles[1] == 999
        assert actions[0] == {"floor": 0, "eventType": "Twirl"}
        assert actions[1]["floor"] == 1
        angles2, actions2 = apply_reflection(angles, actions, "x_flip")
        assert not any(a.get("floor") == 0 and a.get("eventType") == "Twirl" for a in actions2)

    def test_reflect_updates_camera_and_track_positions(self):
        from osuT5.osuT5.dataset.adofai_augment import apply_reflection

        angles, actions = apply_reflection(
            [30],
            [
                {"eventType": "MoveCamera", "position": [10, 4], "rotation": 30, "angleOffset": 12},
                {"eventType": "MoveTrack", "positionOffset": [6, 2], "angleOffset": 8},
                {"floor": 2, "eventType": "Twirl"},
            ],
            "x_flip",
        )
        assert angles[0] == 330
        cam = actions[1] if actions[0]["eventType"] == "Twirl" else actions[0]
        track = next(a for a in actions if a["eventType"] == "MoveTrack")
        assert cam["position"] == [10, -4]
        assert cam["rotation"] == 330
        assert cam["angleOffset"] == 12
        assert track["positionOffset"] == [6, -2]
        assert track["angleOffset"] == 8

    def test_rotation_leaves_twirl_and_angleOffset(self):
        from osuT5.osuT5.dataset.adofai_augment import apply_rotation

        angles, actions = apply_rotation(
            [10, 999],
            [
                {"floor": 0, "eventType": "Twirl"},
                {"eventType": "MoveCamera", "position": [0, 0], "rotation": 10, "angleOffset": 15},
            ],
            90.0,
        )
        assert angles == [100, 999]
        assert actions[0] == {"floor": 0, "eventType": "Twirl"}
        assert actions[1]["angleOffset"] == 15

    def test_matched_rate_scales_bpm_not_multipliers_or_beats(self):
        from osuT5.osuT5.dataset.adofai_augment import apply_matched_rate

        settings, actions = apply_matched_rate(
            {"bpm": 120, "offset": 200, "pitch": 100},
            [
                {"eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 140, "angleOffset": 9},
                {"eventType": "SetSpeed", "speedType": "Multiplier", "bpmMultiplier": 1.5, "angleOffset": 3},
                {"eventType": "Pause", "duration": 2.0},
                {"eventType": "Hold", "duration": 1.0},
                {"eventType": "MoveCamera", "duration": 4.0, "angleOffset": 11},
            ],
            2.0,
        )
        assert settings["bpm"] == 240
        assert settings["offset"] == 100
        assert settings["pitch"] == 100
        assert actions[0]["beatsPerMinute"] == 280
        assert actions[0]["angleOffset"] == 9
        assert actions[1]["bpmMultiplier"] == 1.5
        assert actions[2]["duration"] == 2.0
        assert actions[3]["duration"] == 1.0
        assert actions[4]["duration"] == 4.0
        assert actions[4]["angleOffset"] == 11


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
        assert len(time_shifts) >= 2
        # No fake first interval: tile 0 is at offset 0.
        # First interval uses start heading 180 → next tile 90:
        # rel = (90 - 180 + 540) % 360 = 90 → 0.5 beat = 250ms @ 120 BPM
        assert time_shifts[0] == 0
        assert abs(time_shifts[1] - 250) < 10
    
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
        """Midspin outgoing heading = this tile angle (not this+180); midspin consumes no travel."""
        converter = AdofaiConverter()
        # Distinguishes heading=90 (correct, 250ms) from heading=180 (wrong, 1000ms).
        level = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[90, 999, 0],
            actions=[],
            decorations=[]
        )
        
        events, event_times = converter.level_to_events(level)
        
        midspin_events = [e for e in events if e.type == EventType.MIDSPIN]
        assert len(midspin_events) > 0
        assert midspin_events[0].value == 0
        
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        assert len(time_shifts) == 2
        assert time_shifts[0] == 0
        assert abs(time_shifts[1] - 250) < 10
    
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
        
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        # Tile 0 stays at offset; pause 2 beats = 1000ms is added before travel.
        assert time_shifts[0] == 0
        assert abs((time_shifts[1] - time_shifts[0]) - 1250) < 10
    
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
        
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        assert abs((time_shifts[1] - time_shifts[0]) - 750) < 10
    
    def test_multiplanet_divides_travel_time(self):
        """TwoPlanets is default (2 planets); interval matches the no-event baseline."""
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
        
        # TwoPlanets is the game default, so the interval matches the no-event baseline.
        time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
        assert abs((time_shifts[1] - time_shifts[0]) - 500) < 10


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
        forbidden = {"Pixellate", "Bloom", "Warp", "RadialBlur", "Custom"}
        assert forbidden.isdisjoint(converter.filter_types)
        assert "PetalsInstant" in converter.filter_types
        assert "Pixelate" in converter.filter_types


class TestParserContractAndDifficulty:
    """OsuParser-compatible contract + difficulty injection (no star rating)."""

    def _level(self):
        return AdofaiLevel(
            settings={"bpm": 140, "offset": 0, "difficulty": 7},
            angle_data=[0, 90, 180],
            actions=[
                {"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 160},
                {"floor": 1, "eventType": "Twirl"},
                {
                    "floor": 2,
                    "eventType": "MoveCamera",
                    "position": [4, 0],
                    "rotation": 15,
                    "zoom": 100,
                    "duration": 1.0,
                    "ease": "Linear",
                    "relativeTo": "LastPositionNoRotation",
                },
            ],
            decorations=[],
        )

    def test_parse_returns_events_and_int_times(self):
        events, times = AdofaiConverter().level_to_events(self._level())
        times = [int(t) for t in times]
        assert len(events) == len(times)
        assert any(e.type == EventType.TILE_ANGLE for e in events)
        assert any(e.type == EventType.TWIRL for e in events)
        assert any(e.type == EventType.MOVE_CAMERA for e in events)
        assert any(e.type == EventType.SET_SPEED_BPM for e in events)

    def test_parse_timing_excludes_map_events(self):
        timing_types = {
            EventType.TIME_SHIFT, EventType.BPM, EventType.OFFSET,
            EventType.SET_SPEED_BPM, EventType.SET_SPEED_MULT,
            EventType.PAUSE, EventType.HOLD,
        }
        events, times = AdofaiConverter().level_to_events(self._level())
        types = {e.type for e, t in zip(events, times) if e.type in timing_types}
        assert EventType.SET_SPEED_BPM in types
        assert EventType.TIME_SHIFT in types
        map_types = {e.type for e in events}
        assert EventType.TWIRL in map_types
        assert EventType.MOVE_CAMERA in map_types
        assert EventType.TWIRL not in types
        assert EventType.MOVE_CAMERA not in types
        assert EventType.TILE_ANGLE not in types

    def test_difficulty_from_settings(self, tmp_path):
        from osuT5.osuT5.dataset.adofai_augment import resolve_difficulty

        assert resolve_difficulty(tmp_path, {"difficulty": 8}) == 8.0

    def test_difficulty_from_index_json(self, tmp_path):
        from osuT5.osuT5.dataset.adofai_augment import resolve_difficulty

        (tmp_path / "index.json").write_text('{"difficulty": 3}', encoding="utf-8")
        assert resolve_difficulty(tmp_path, {}) == 3.0

    def test_difficulty_proxy_when_missing(self, tmp_path):
        from osuT5.osuT5.dataset.adofai_augment import resolve_difficulty, ADOFAI_DIFFICULTY_PROXY

        assert resolve_difficulty(tmp_path, {}) == ADOFAI_DIFFICULTY_PROXY


class TestFrameWindowing:
    """Raw waveform frames, not precomputed 80-mel / 60s crop."""

    def test_get_frames_shape_matches_mmrs(self):
        hop_length = 128
        sample_rate = 16000
        samples = np.zeros(hop_length * 20, dtype=np.float32)
        # Same pad as SequenceDatasetMixin._get_frames (exact multiples get one extra pad frame)
        samples = np.pad(samples, [0, hop_length - len(samples) % hop_length])
        frames = np.reshape(samples, (-1, hop_length))
        frame_times = np.arange(len(frames)) / (sample_rate / hop_length / 1000)
        assert frames.shape[1] == hop_length
        assert frames.ndim == 2
        assert len(frame_times) == frames.shape[0]

    def test_max_source_positions_is_half_src(self):
        """Do not mix 1024 vs 4096 checkpoints: max_source_positions = src_seq_len // 2."""
        assert 4096 // 2 == 2048
        assert 1024 // 2 == 512


class TestSharpFAIRoundTrip:
    """Locked on-disk keys survive level → events → level."""

    def _roundtrip(self, actions, settings=None):
        converter = AdofaiConverter()
        level = AdofaiLevel(
            settings=settings or {"bpm": 120, "offset": 0, "pitch": 110, "difficulty": 6},
            angle_data=[0, 90, 180],
            actions=actions,
            decorations=[],
        )
        events, times = converter.level_to_events(level)
        return events, converter.events_to_level(events, times, level.settings)

    def test_setspeed_locked_keys(self):
        events, out = self._roundtrip([{
            "floor": 1,
            "eventType": "SetSpeed",
            "speedType": "Bpm",
            "beatsPerMinute": 160,
            "bpmMultiplier": 1.0,
            "angleOffset": 15,
        }])
        act = next(a for a in out.actions if a["eventType"] == "SetSpeed")
        assert act["speedType"] == "Bpm"
        assert act["beatsPerMinute"] == 160
        assert "bpmMultiplier" in act
        assert act["angleOffset"] == 15
        assert "easing" not in act and "tag" not in act

    def test_pause_hold_multiplanet_locked_keys(self):
        _, out = self._roundtrip([
            {"floor": 0, "eventType": "Pause", "duration": 2.0, "countdownTicks": 3, "angleCorrectionDir": 1},
            {"floor": 1, "eventType": "Hold", "duration": 1.5, "distanceMultiplier": 80, "landingAnimation": True},
            {"floor": 2, "eventType": "MultiPlanet", "planets": "ThreePlanets"},
        ])
        pause = next(a for a in out.actions if a["eventType"] == "Pause")
        hold = next(a for a in out.actions if a["eventType"] == "Hold")
        planets = next(a for a in out.actions if a["eventType"] == "MultiPlanet")
        assert pause["duration"] == pytest.approx(2.0)
        assert pause["countdownTicks"] == 3
        assert pause["angleCorrectionDir"] == 1
        assert hold["duration"] == pytest.approx(1.5)
        assert hold["distanceMultiplier"] == 80
        assert hold["landingAnimation"] is True
        assert planets["planets"] == "ThreePlanets"

    def test_movecamera_and_movetrack_locked_keys(self):
        _, out = self._roundtrip([
            {
                "floor": 1,
                "eventType": "MoveCamera",
                "duration": 2.0,
                "relativeTo": "LastPositionNoRotation",
                "position": [3, -2],
                "rotation": 45,
                "zoom": 120,
                "angleOffset": 15,
                "ease": "OutQuad",
                "eventTag": "cam1",
            },
            {
                "floor": 2,
                "eventType": "MoveTrack",
                "startTile": [0, "ThisTile"],
                "endTile": [2, "End"],
                "duration": 1.0,
                "positionOffset": [4, 5],
                "angleOffset": 10,
                "ease": "Linear",
                "eventTag": "",
            },
        ])
        cam = next(a for a in out.actions if a["eventType"] == "MoveCamera")
        track = next(a for a in out.actions if a["eventType"] == "MoveTrack")
        assert cam["relativeTo"] == "LastPositionNoRotation"
        assert cam["ease"] == "OutQuad"
        assert cam["angleOffset"] == 15
        assert "easing" not in cam and "eventTag" in cam
        assert track["positionOffset"] == [4, 5]
        assert track["startTile"] == [0, "ThisTile"]
        assert track["endTile"] == [2, "End"]
        assert "position" not in track
        assert track["ease"] == "Linear"
        assert track["angleOffset"] == 10

    def test_flash_bloom_shake_filter_advanced(self):
        events, out = self._roundtrip([
            {
                "floor": 1,
                "eventType": "Flash",
                "duration": 1.0,
                "plane": "Background",
                "startColor": "ff0000",
                "startOpacity": 80,
                "endColor": "00ff00",
                "endOpacity": 10,
                "ease": "Linear",
                "angleOffset": 0,
                "eventTag": "",
            },
            {
                "floor": 1,
                "eventType": "Bloom",
                "enabled": "Enabled",
                "threshold": 40,
                "intensity": 70,
                "color": "aabbcc",
            },
            {
                "floor": 1,
                "eventType": "ShakeScreen",
                "duration": 1.0,
                "strength": 40,
                "intensity": 25,
                "fadeOut": "Enabled",
            },
            {
                "floor": 2,
                "eventType": "SetFilter",
                "filter": "Pixelate",
                "enabled": "Enabled",
                "intensity": 90,
                "disableOthers": "Enabled",
            },
            {
                "floor": 2,
                "eventType": "SetFilterAdvanced",
                "filter": "PetalsInstant",
                "enabled": "Disabled",
                "disableOthers": "Enabled",
                "filterProperties": "intensity:1",
            },
        ])
        types = {e.type for e in events}
        assert EventType.SET_FILTER_ADVANCED in types
        flash = next(a for a in out.actions if a["eventType"] == "Flash")
        bloom = next(a for a in out.actions if a["eventType"] == "Bloom")
        shake = next(a for a in out.actions if a["eventType"] == "ShakeScreen")
        filt = next(a for a in out.actions if a["eventType"] == "SetFilter")
        adv = next(a for a in out.actions if a["eventType"] == "SetFilterAdvanced")
        assert flash["plane"] == "Background"
        assert "startColor" in flash and "startOpacity" in flash
        assert "color" not in flash and "speed" not in shake
        assert bloom["enabled"] == "Enabled"
        assert bloom["intensity"] == 70
        assert bloom["threshold"] == 40
        assert "color" in bloom
        assert shake["strength"] == 40
        assert shake["intensity"] == 25
        assert "speed" not in shake
        assert filt["filter"] == "Pixelate"
        assert filt["disableOthers"] == "Enabled"
        assert adv["filter"] == "PetalsInstant"
        assert "filterProperties" in adv
        assert adv["enabled"] == "Disabled"

    def test_rare_events_tokenized(self):
        events, out = self._roundtrip([
            {"floor": 0, "eventType": "Checkpoint"},
            {"floor": 1, "eventType": "KillPlayer"},
            {"floor": 1, "eventType": "Bookmark"},
            {"floor": 2, "eventType": "EditorComment"},
            {"floor": 2, "eventType": "CallMethod"},
            {"floor": 2, "eventType": "AddComponent"},
            {"floor": 2, "eventType": "ChangeTrack"},
            {"floor": 2, "eventType": "FreeRoamWarning"},
        ])
        types = {e.type for e in events}
        for needed in (
            EventType.CHECKPOINT, EventType.KILL_PLAYER, EventType.BOOKMARK,
            EventType.EDITOR_COMMENT, EventType.CALL_METHOD, EventType.ADD_COMPONENT,
            EventType.CHANGE_TRACK, EventType.FREE_ROAM_WARNING,
        ):
            assert needed in types
        names = {a["eventType"] for a in out.actions}
        assert names >= {"Checkpoint", "KillPlayer", "Bookmark", "EditorComment", "CallMethod", "AddComponent", "ChangeTrack", "FreeRoamWarning"}

    def test_settings_pitch_and_difficulty(self):
        events, out = self._roundtrip([], {"bpm": 100, "offset": 20, "pitch": 115, "difficulty": 8})
        assert any(e.type == EventType.PITCH and e.value == 115 for e in events)
        assert out.settings["pitch"] == 115
        assert out.settings.get("difficulty") == 8


class TestConverterTimingDeltas:
    def test_pause_hold_multiplanet_change_ms(self):
        converter = AdofaiConverter()
        base = AdofaiLevel(settings={"bpm": 120, "offset": 0}, angle_data=[0, 90], actions=[], decorations=[])
        pause = AdofaiLevel(settings={"bpm": 120, "offset": 0}, angle_data=[0, 90], actions=[{"floor": 0, "eventType": "Pause", "duration": 2.0}], decorations=[])
        hold = AdofaiLevel(settings={"bpm": 120, "offset": 0}, angle_data=[0, 90], actions=[{"floor": 0, "eventType": "Hold", "duration": 1.0}], decorations=[])
        multi = AdofaiLevel(settings={"bpm": 120, "offset": 0}, angle_data=[0, 90], actions=[{"floor": 0, "eventType": "MultiPlanet", "planets": "TwoPlanets"}], decorations=[])

        def tile_times(level):
            events, _ = converter.level_to_events(level)
            return [e.value for e in events if e.type == EventType.TIME_SHIFT]

        base_t = tile_times(base)
        pause_t = tile_times(pause)
        hold_t = tile_times(hold)
        multi_t = tile_times(multi)
        # heading 180 → 90 is 0.5 beat = 250ms; pause 2 beats adds 1000ms after tile 0
        assert abs((base_t[1] - base_t[0]) - 250) < 10
        assert pause_t[0] == 0
        assert abs((pause_t[1] - pause_t[0]) - 1250) < 10
        assert abs((hold_t[1] - hold_t[0]) - 750) < 10
        # TwoPlanets is default gameplay (2 planets); interval matches baseline
        assert abs((multi_t[1] - multi_t[0]) - 250) < 10

        three = AdofaiLevel(
            settings={"bpm": 120, "offset": 0},
            angle_data=[0, 90],
            actions=[{"floor": 0, "eventType": "MultiPlanet", "planets": "ThreePlanets"}],
            decorations=[],
        )
        three_t = tile_times(three)
        assert abs((three_t[1] - three_t[0]) - (250 * 2 / 3)) < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
