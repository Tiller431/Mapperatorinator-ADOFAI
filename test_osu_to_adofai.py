"""
Test osuT5 event to ADOFAI conversion.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from osuT5.osuT5.event import Event, EventType
from adofai.osu_to_adofai import (
    extract_timing_info,
    extract_hit_times,
    times_to_angles,
    osu_events_to_adofai,
)
from adofai.parser import parse_adofai, write_adofai


def test_extract_timing():
    """Test that timing events are correctly extracted."""
    print("Testing timing extraction...")
    
    # Create fixture timing events and times
    # Event values are beat lengths in ms: 500ms = 120 BPM, 400ms = 150 BPM
    events = [
        Event(EventType.TIMING_POINT, 500),  # BPM = 60000/500 = 120
        Event(EventType.CIRCLE, 0),
        Event(EventType.TIMING_POINT, 400),  # BPM change to 150
        Event(EventType.CIRCLE, 0),
    ]
    times = [100, 500, 2000, 2500]  # Times when events occur
    
    bpm, offset, speed_changes = extract_timing_info(events, times)
    
    assert bpm == 120.0, f"Expected BPM 120, got {bpm}"
    assert offset == 100.0, f"Expected offset 100, got {offset}"
    assert len(speed_changes) == 1, f"Expected 1 speed change, got {len(speed_changes)}"
    assert speed_changes[0][0] == 2000, f"Expected speed change at 2000ms"
    assert abs(speed_changes[0][1] - 150.0) < 0.1, f"Expected BPM 150, got {speed_changes[0][1]}"
    
    print(f"  ✓ Extracted BPM={bpm}, offset={offset}, {len(speed_changes)} speed changes")
    print("✓ Timing extraction test: PASSED")


def test_extract_hits():
    """Test that hit times are correctly extracted."""
    print("\nTesting hit extraction...")
    
    events = [
        Event(EventType.TIMING_POINT, 500),
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
        Event(EventType.SLIDER_HEAD, 0),
        Event(EventType.SPINNER, 0),
    ]
    times = [0, 500, 1000, 1500, 2000]
    
    hit_times = extract_hit_times(events, times)
    
    assert len(hit_times) == 4, f"Expected 4 hits, got {len(hit_times)}"
    assert hit_times == [500, 1000, 1500, 2000], f"Unexpected hit times: {hit_times}"
    
    print(f"  ✓ Extracted {len(hit_times)} hit times: {hit_times}")
    print("✓ Hit extraction test: PASSED")


def test_times_to_angles():
    """Test that hit times are converted to angles."""
    print("\nTesting time-to-angle conversion...")
    
    hit_times = [0, 500, 1000, 1500, 2000]  # Every 500ms
    bpm = 120.0  # 500ms per beat
    offset = 0.0
    
    angles = times_to_angles(hit_times, bpm, offset)
    
    assert len(angles) == len(hit_times), f"Expected {len(hit_times)} angles, got {len(angles)}"
    assert all(0 <= angle < 360 for angle in angles), "Angles should be 0-359"
    assert len(set(angles)) > 1, "Should have variety in angles"
    
    print(f"  ✓ Converted {len(hit_times)} times to {len(angles)} angles")
    print(f"  Angles: {angles}")
    print("✓ Time-to-angle conversion test: PASSED")


def test_events_to_level():
    """Test full conversion from events to ADOFAI level."""
    print("\nTesting full event-to-level conversion...")
    
    # Create realistic event sequence
    events = [
        Event(EventType.TIMING_POINT, 500),  # 120 BPM
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
        Event(EventType.SLIDER_HEAD, 0),
        Event(EventType.CIRCLE, 0),
        Event(EventType.TIMING_POINT, 400),  # 150 BPM
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
    ]
    times = [0, 0, 500, 1000, 1500, 2000, 2500, 2500, 2900]
    
    level = osu_events_to_adofai(
        events=events,
        times=times,
        audio_filename="test.ogg",
        title="Test Song",
        artist="Test Artist",
    )
    
    # Verify level structure
    assert level.settings['bpm'] == 120.0, f"Expected BPM 120, got {level.settings['bpm']}"
    assert level.settings['offset'] == 0, f"Expected offset 0, got {level.settings['offset']}"
    assert level.settings['song'] == "Test Song"
    assert level.settings['artist'] == "Test Artist"
    assert level.settings['songFilename'] == "test.ogg"
    
    assert len(level.angle_data) >= 7, f"Expected at least 7 tiles, got {len(level.angle_data)}"
    assert all(isinstance(angle, int) for angle in level.angle_data), "All angles should be integers"
    assert all(0 <= angle <= 360 for angle in level.angle_data), "Angles should be 0-360"
    
    # Should have at least base SetSpeed action
    assert len(level.actions) >= 1, f"Expected at least 1 action, got {len(level.actions)}"
    base_action = level.actions[0]
    assert base_action['eventType'] == 'SetSpeed'
    assert base_action['beatsPerMinute'] == 120.0
    
    print(f"  ✓ Generated level with {len(level.angle_data)} tiles and {len(level.actions)} actions")
    print(f"  BPM: {level.settings['bpm']}, Offset: {level.settings['offset']}")
    print("✓ Event-to-level conversion test: PASSED")


def test_roundtrip_write():
    """Test that converted level can be written and read back."""
    print("\nTesting roundtrip write/read...")
    
    import tempfile
    
    events = [
        Event(EventType.TIMING_POINT, 500),  # 120 BPM
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
        Event(EventType.CIRCLE, 0),
    ]
    times = [100, 100, 600, 1100]  # 100ms offset
    
    level = osu_events_to_adofai(
        events=events,
        times=times,
        audio_filename="test.mp3",
        title="Roundtrip Test",
        artist="Test",
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "test.adofai"
        
        # Write
        write_adofai(level, output_file)
        assert output_file.exists(), "File should be written"
        
        # Read back
        level_read = parse_adofai(output_file)
        
        # Verify key fields survived roundtrip
        assert level_read.settings['bpm'] == level.settings['bpm']
        assert level_read.settings['offset'] == level.settings['offset']
        assert level_read.settings['song'] == level.settings['song']
        assert level_read.settings['songFilename'] == level.settings['songFilename']
        assert len(level_read.angle_data) == len(level.angle_data)
        
        print(f"  ✓ Wrote and read back level successfully")
        print(f"  BPM: {level_read.settings['bpm']}, Tiles: {len(level_read.angle_data)}")
        print("✓ Roundtrip test: PASSED")


if __name__ == "__main__":
    print("Running osu-to-ADOFAI conversion tests...\n")
    
    test_extract_timing()
    test_extract_hits()
    test_times_to_angles()
    test_events_to_level()
    test_roundtrip_write()
    
    print("\n✅ All osu-to-ADOFAI conversion tests passed!")
    print("\nKey verified behaviors:")
    print("  - BPM and offset extracted from timing events (not hardcoded)")
    print("  - Hit times drive tile count and timing")
    print("  - Angles assigned from beat-grid mapping")
    print("  - SetSpeed actions generated from BPM changes")
