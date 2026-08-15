"""
DEPRECATED: osu → ADOFAI event conversion stub.

This file attempts to convert osu! events to ADOFAI, which is NOT the production inference path.

Production inference path:
    1. Train a Whisper model on ADOFAI data: python osuT5/train.py -cn adofai_v31
    2. Generate ADOFAI events directly using the trained checkpoint
    3. Convert osuT5 Events → .adofai using adofai/converter.py events_to_level()

This file's "osu hit times → ADOFAI beat-grid angles" approach does NOT produce
valid ADOFAI charts and should NOT be used.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from osuT5.event import Event, EventType
except ModuleNotFoundError:
    from osuT5.osuT5.event import Event, EventType
from adofai.event import AdofaiEvent, AdofaiEventType
from adofai.parser import AdofaiLevel, write_adofai


def extract_timing_info(events: list[Event], times: list[float]) -> tuple[float, float, list[tuple[float, float]]]:
    """
    Extract BPM, offset, and speed changes from osu timing events.
    
    Args:
        events: List of Event objects
        times: Corresponding timestamps in milliseconds
    
    Returns:
        bpm: Base BPM from first timing point
        offset: Offset in milliseconds
        speed_changes: List of (time_ms, bpm) for SetSpeed actions
    """
    bpm = 120.0  # Default
    offset = 0.0
    speed_changes = []
    found_first = False
    
    for event, time in zip(events, times):
        if event.type == EventType.TIMING_POINT:
            # First timing point sets base BPM and offset
            if event.value > 0:  # Positive value = BPM timing point
                event_bpm = 60000.0 / event.value  # Convert beat length to BPM
                if not found_first:  # First one
                    bpm = event_bpm
                    offset = time
                    found_first = True
                else:
                    speed_changes.append((time, event_bpm))
    
    return bpm, offset, speed_changes


def extract_hit_times(events: list[Event], times: list[float]) -> list[float]:
    """
    Extract hit/note times from osu events.
    
    Args:
        events: List of Event objects
        times: Corresponding timestamps in milliseconds
    
    Returns list of timestamps in milliseconds where tiles should appear.
    """
    hit_times = []
    
    # All these event types represent a hit/action that should become a tile
    hit_event_types = {
        EventType.CIRCLE,
        EventType.SLIDER_HEAD,
        EventType.SPINNER,
        EventType.HOLD_NOTE,
        EventType.DRUMROLL,
        EventType.DENDEN,
    }
    
    for event, time in zip(events, times):
        if event.type in hit_event_types:
            hit_times.append(time)
    
    return sorted(set(hit_times))  # Unique and sorted


def times_to_angles(hit_times: list[float], bpm: float, offset: float) -> list[int]:
    """
    Convert hit times to ADOFAI tile angles.
    
    Strategy: Map hits to a beat grid, then assign angles in a musically-timed pattern.
    Uses a small discrete set of angles (0, 45, 90, 135, 180, 225, 270, 315)
    to create playable patterns.
    
    Args:
        hit_times: List of hit timestamps in ms
        bpm: Beats per minute
        offset: Timing offset in ms
    
    Returns:
        List of tile angles (0-359 degrees)
    """
    if not hit_times:
        # Return a minimal playable pattern
        return [0, 0, 90, 180]
    
    # Common angles for playable patterns
    angle_set = [0, 45, 90, 135, 180, 225, 270, 315]
    
    ms_per_beat = 60000.0 / bpm
    angles = []
    
    for i, hit_time in enumerate(hit_times):
        # Calculate which beat this hit is on (relative to offset)
        beat_position = (hit_time - offset) / ms_per_beat
        
        # Use beat position to pick angle
        # Every 4 beats, cycle through the angle set
        angle_index = int(beat_position) % len(angle_set)
        angle = angle_set[angle_index]
        
        # Add some variation on off-beats
        if beat_position % 1.0 > 0.4:  # Off-beat hit
            # Rotate 45 degrees
            angle = (angle + 45) % 360
        
        angles.append(angle)
    
    # Ensure we have at least a few tiles
    if len(angles) < 3:
        angles.extend([0, 90, 180])
    
    return angles


def osu_events_to_adofai(
    events: list[Event],
    times: list[float],
    audio_filename: str,
    title: str = "Generated Song",
    artist: str = "Unknown Artist",
    creator: str = "Mapperatorinator ADOFAI",
) -> AdofaiLevel:
    """
    Convert osuT5 event stream to ADOFAI level.
    
    Args:
        events: List of osuT5 Event objects from processor.generate()
        times: Corresponding timestamps in milliseconds
        audio_filename: Name of audio file
        title: Song title
        artist: Artist name
        creator: Chart creator
    
    Returns:
        AdofaiLevel ready to write
    """
    # Extract timing information
    bpm, offset, speed_changes = extract_timing_info(events, times)
    
    # Extract hit times
    hit_times = extract_hit_times(events, times)
    
    # Convert hit times to angles
    angle_data = times_to_angles(hit_times, bpm, offset)
    
    # Build actions from speed changes
    actions = []
    
    # Add base SetSpeed action
    actions.append({
        "floor": 1,
        "eventType": "SetSpeed",
        "speedType": "Bpm",
        "beatsPerMinute": bpm,
        "bpmMultiplier": 1.0
    })
    
    # Add speed changes if any
    for time_ms, new_bpm in speed_changes:
        # Find which floor (tile) this corresponds to
        # For now, map to approximate floor based on time
        floor_index = 1 + int((time_ms - offset) / (60000.0 / bpm) / 0.5)  # Rough estimate
        floor_index = max(2, min(floor_index, len(angle_data)))
        
        actions.append({
            "floor": floor_index,
            "eventType": "SetSpeed",
            "speedType": "Bpm",
            "beatsPerMinute": new_bpm,
            "bpmMultiplier": 1.0
        })
    
    # Add occasional twirl for variety (every ~16 tiles)
    if len(angle_data) > 16:
        actions.append({
            "floor": len(angle_data) // 2,
            "eventType": "Twirl"
        })
    
    # Build settings
    settings = {
        "version": 14,
        "artist": artist,
        "song": title,
        "author": creator,
        "separateCountdownTime": True,
        "previewSongStart": 0,
        "previewSongDuration": 10,
        "seizureWarning": False,
        "levelDesc": f"Generated by Mapperatorinator ADOFAI from osuT5 pipeline (BPM: {bpm:.1f})",
        "levelTags": "ai-generated",
        "artistPermission": "",
        "songFilename": audio_filename,
        "bpm": bpm,
        "volume": 100,
        "offset": int(offset),
        "pitch": 100,
        "hitsound": "Kick",
        "hitsoundVolume": 100,
    }
    
    return AdofaiLevel(
        settings=settings,
        angle_data=angle_data,
        actions=actions,
        decorations=[]
    )


def export_osu_to_adofai(
    events: list[Event],
    times: list[float],
    output_path: str | Path,
    audio_filename: str,
    title: str = "Generated Song",
    artist: str = "Unknown Artist",
    creator: str = "Mapperatorinator ADOFAI",
) -> Path:
    """
    Export osuT5 events to .adofai file.
    
    Args:
        events: osuT5 event stream
        times: Corresponding timestamps in milliseconds
        output_path: Directory to write output
        audio_filename: Audio file name
        title: Song title
        artist: Artist name
        creator: Creator name
    
    Returns:
        Path to written .adofai file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert events to ADOFAI level
    level = osu_events_to_adofai(
        events=events,
        times=times,
        audio_filename=audio_filename,
        title=title,
        artist=artist,
        creator=creator,
    )
    
    # Write to file
    output_file = output_path / f"{title.replace(' ', '_')}.adofai"
    write_adofai(level, output_file)
    
    return output_file
