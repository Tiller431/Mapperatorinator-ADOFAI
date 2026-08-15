"""
Tests for ADOFAI module functionality.

Includes:
- Parser round-trip tests
- Event conversion tests
- Legacy pathData compatibility tests
"""

import os
import json
import tempfile
from pathlib import Path

from adofai import parse_adofai, write_adofai, AdofaiLevel
from adofai.parser import path_data_to_angle_data, angle_data_to_path_data
from adofai.converter import AdofaiConverter


def test_parse_sample_fixture():
    """Test parsing the sample fixture file."""
    print("Testing parse_sample_fixture...")
    
    fixture_path = Path("fixtures/sample.adofai")
    level = parse_adofai(fixture_path)
    
    assert level.settings["bpm"] == 120
    assert level.settings["artist"] == "Test Artist"
    assert len(level.angle_data) == 10
    assert level.angle_data[0] == 0
    assert level.angle_data[8] == 999  # midspin
    assert len(level.actions) == 4
    
    print("✓ Parse sample fixture: PASSED")


def test_roundtrip():
    """Test round-trip: parse → write → parse."""
    print("Testing roundtrip...")
    
    fixture_path = Path("fixtures/sample.adofai")
    level1 = parse_adofai(fixture_path)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.adofai', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        write_adofai(level1, tmp_path)
        level2 = parse_adofai(tmp_path)
        
        # Compare key fields
        assert level1.settings == level2.settings
        assert level1.angle_data == level2.angle_data
        assert len(level1.actions) == len(level2.actions)
        
        # Verify it's valid JSON
        with open(tmp_path, 'r') as f:
            data = json.load(f)
            assert "angleData" in data
            assert "settings" in data
            assert "actions" in data
        
        print("✓ Roundtrip: PASSED")
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_path_data_conversion():
    """Test legacy pathData to angleData conversion."""
    print("Testing path_data_conversion...")
    
    path_data = "RRULDR!"
    angle_data = path_data_to_angle_data(path_data)
    
    expected = [0, 0, 90, 180, 270, 0, 999]
    assert angle_data == expected
    
    # Test reverse conversion
    path_data_back = angle_data_to_path_data(angle_data)
    assert path_data_back == path_data
    
    print("✓ PathData conversion: PASSED")


def test_trailing_comma_tolerance():
    """Test that parser handles trailing commas."""
    print("Testing trailing_comma_tolerance...")
    
    # Create a file with trailing commas
    content = """{
    "settings": {
        "bpm": 100,
        "offset": 0,
        "songFilename": "test.ogg",
    },
    "angleData": [0, 90, 180,],
    "actions": [],
    "decorations": [],
}"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.adofai', delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        level = parse_adofai(tmp_path)
        assert level.settings["bpm"] == 100
        assert level.angle_data == [0, 90, 180]
        
        print("✓ Trailing comma tolerance: PASSED")
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_utf8_bom_handling():
    """Test that parser handles UTF-8 BOM (common in Workshop files)."""
    print("Testing utf8_bom_handling...")
    
    # Create a file with UTF-8 BOM at the start (like Steam Workshop files)
    content = '\ufeff{\n    "settings": {\n        "bpm": 140,\n        "offset": 0,\n        "songFilename": "test.ogg"\n    },\n    "angleData": [0, 90, 180, 270],\n    "actions": [],\n    "decorations": []\n}'
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.adofai', delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        # Should parse successfully despite BOM
        level = parse_adofai(tmp_path)
        assert level.settings["bpm"] == 140
        assert level.angle_data == [0, 90, 180, 270]
        
        print("✓ UTF-8 BOM handling: PASSED")
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_event_conversion():
    """Test converting level to events and back."""
    print("Testing event_conversion...")
    
    fixture_path = Path("fixtures/sample.adofai")
    level = parse_adofai(fixture_path)
    
    converter = AdofaiConverter()
    
    # Convert to events
    events, event_times = converter.level_to_events(level)
    
    assert len(events) > 0
    assert len(events) == len(event_times)
    
    # Verify some expected events
    event_types = [e.type.value for e in events]
    assert "bpm" in event_types
    assert "angle" in event_types
    assert "twirl" in event_types
    assert "midspin" in event_types
    
    # Convert back to level
    level2 = converter.events_to_level(events, event_times, base_settings=level.settings)
    
    assert level2.settings["bpm"] == level.settings["bpm"]
    assert len(level2.angle_data) > 0
    
    print("✓ Event conversion: PASSED")


def test_create_minimal_level():
    """Test creating a minimal valid level from scratch."""
    print("Testing create_minimal_level...")
    
    level = AdofaiLevel(
        settings={
            "version": 14,
            "artist": "Test",
            "song": "Test Song",
            "author": "Mapperatorinator",
            "songFilename": "test.ogg",
            "bpm": 140,
            "offset": 0,
            "volume": 100,
            "pitch": 100,
            "hitsound": "Kick",
            "hitsoundVolume": 100,
        },
        angle_data=[0, 0, 90, 180, 270],
        actions=[
            {
                "floor": 1,
                "eventType": "SetSpeed",
                "speedType": "Bpm",
                "beatsPerMinute": 140,
                "bpmMultiplier": 1
            }
        ]
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.adofai', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        write_adofai(level, tmp_path)
        
        # Verify it can be parsed back
        level2 = parse_adofai(tmp_path)
        assert level2.settings["bpm"] == 140
        assert level2.angle_data == [0, 0, 90, 180, 270]
        
        print("✓ Create minimal level: PASSED")
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    print("Running ADOFAI module tests...\n")
    
    test_parse_sample_fixture()
    test_roundtrip()
    test_path_data_conversion()
    test_trailing_comma_tolerance()
    test_utf8_bom_handling()
    test_event_conversion()
    test_create_minimal_level()
    
    print("\n✅ All tests passed!")
