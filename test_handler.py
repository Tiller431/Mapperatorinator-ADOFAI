#!/usr/bin/env python3
"""
Test script for the RunPod handler
"""

import base64
import json
import os
from pathlib import Path

# Import the handler
from handler import runpod_handler


def test_handler_minimal():
    """Test handler with minimal input (no audio, just validation)"""
    test_event = {
        "input": {
            "title": "Test Song",
            "artist": "Test Artist",
            "temperature": 0.9,
            "cfg_scale": 1.0
        }
    }
    
    result = runpod_handler(test_event)
    print("Minimal test result:")
    print(json.dumps(result, indent=2))
    
    # Should return an error about missing audio
    assert "error" in result
    assert "audio" in result["error"]
    print("✓ Minimal test passed (correctly detected missing audio)")


def test_handler_with_audio(audio_file_path: str):
    """Test handler with actual audio file"""
    if not Path(audio_file_path).exists():
        print(f"⚠️  Audio file not found: {audio_file_path}")
        print("   Skipping audio test")
        return
    
    # Read and encode audio file
    with open(audio_file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    test_event = {
        "input": {
            "audio": audio_data,
            "title": "Test Song",
            "artist": "Test Artist", 
            "creator": "Test Mapper",
            "version": "Test Diff",
            "gamemode": 0,
            "temperature": 0.9,
            "cfg_scale": 1.0,
            "top_p": 0.9,
            "overall_difficulty": 8.0,
            "approach_rate": 9.0,
            "circle_size": 4.0,
            "hp_drain_rate": 5.0,
            "seed": 12345
        }
    }
    
    print(f"Testing with audio file: {audio_file_path}")
    print("This will take a while as it needs to generate a full beatmap...")
    
    result = runpod_handler(test_event)
    
    if "error" in result:
        print("❌ Audio test failed:")
        print(json.dumps(result, indent=2))
        return False
    
    print("✓ Audio test passed!")
    print(f"Generated file: {result.get('filename', 'unknown')}")
    print(f"Metadata: {result.get('metadata', {})}")
    
    # Save the generated .osz file for inspection
    if "osz_file" in result:
        output_file = f"test_output_{result.get('filename', 'generated.osz')}"
        osz_data = base64.b64decode(result["osz_file"])
        with open(output_file, "wb") as f:
            f.write(osz_data)
        print(f"✓ Saved generated beatmap to: {output_file}")
    
    return True


def test_handler_with_beatmap_reference(audio_file_path: str, beatmap_file_path: str):
    """Test handler with both audio and beatmap reference"""
    if not Path(audio_file_path).exists():
        print(f"⚠️  Audio file not found: {audio_file_path}")
        return
    
    if not Path(beatmap_file_path).exists():
        print(f"⚠️  Beatmap file not found: {beatmap_file_path}")
        return
    
    # Read and encode files
    with open(audio_file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    with open(beatmap_file_path, "r", encoding='utf-8') as f:
        beatmap_data = base64.b64encode(f.read().encode('utf-8')).decode('utf-8')
    
    test_event = {
        "input": {
            "audio": audio_data,
            "beatmap": beatmap_data,
            "temperature": 0.9,
            "cfg_scale": 1.0,
            "seed": 12345
        }
    }
    
    print(f"Testing with audio: {audio_file_path} and beatmap: {beatmap_file_path}")
    
    result = runpod_handler(test_event)
    
    if "error" in result:
        print("❌ Beatmap reference test failed:")
        print(json.dumps(result, indent=2))
        return False
    
    print("✓ Beatmap reference test passed!")
    print(f"Generated file: {result.get('filename', 'unknown')}")
    print(f"Metadata: {result.get('metadata', {})}")
    
    return True


if __name__ == "__main__":
    print("Testing RunPod Handler")
    print("=" * 50)
    
    # Test 1: Minimal test (should fail gracefully)
    print("\n1. Testing minimal input (no audio)...")
    test_handler_minimal()
    
    # Test 2: Test with audio file (if available)
    print("\n2. Testing with audio file...")
    
    # Look for test audio files in common locations
    test_audio_paths = [
        "test_audio.mp3",
        "test_audio.wav", 
        "audio.mp3",
        "sample.mp3",
        "/tmp/test_audio.mp3"
    ]
    
    audio_path = None
    for path in test_audio_paths:
        if Path(path).exists():
            audio_path = path
            break
    
    if audio_path:
        test_handler_with_audio(audio_path)
    else:
        print("⚠️  No test audio file found. Place a test audio file as 'test_audio.mp3' to test audio processing.")
    
    # Test 3: Test with beatmap reference (if available)
    print("\n3. Testing with beatmap reference...")
    
    test_beatmap_paths = [
        "test_beatmap.osu",
        "reference.osu",
        "sample.osu",
        "/tmp/test_beatmap.osu"
    ]
    
    beatmap_path = None
    for path in test_beatmap_paths:
        if Path(path).exists():
            beatmap_path = path
            break
    
    if audio_path and beatmap_path:
        test_handler_with_beatmap_reference(audio_path, beatmap_path)
    else:
        print("⚠️  No test beatmap file found or no audio file. Place a test .osu file as 'test_beatmap.osu' to test beatmap references.")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("\nTo test with your own files:")
    print("  python test_handler.py")
    print("  # Place test_audio.mp3 and test_beatmap.osu in the current directory")
