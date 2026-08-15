"""ADOFAI EventRange tables and encode used by Tokenizer and smoke tests.

Kept torch-free so tests can encode converter output without importing tokenizer.py.
"""

from __future__ import annotations

from .event import Event, EventRange, EventType


def adofai_event_ranges() -> list[EventRange]:
    """Ranges for types ``AdofaiConverter.level_to_events`` emits on the map stream.

    Camera/VFX/gameplay bounds are widened from Tiller727/adofai-charts-v1
    (Workshop ``level.adofai``) plus known smoke overflows (zoom 1000,
    RepeatEvents 48, filter intensity 500). Values are stored as-is; there
    is no clip-to-old-max (1000 must not become 400).
    """
    return [
        EventRange(EventType.TILE_ANGLE, 0, 359),
        EventRange(EventType.MIDSPIN, 0, 0),
        EventRange(EventType.SET_SPEED_BPM, 1, 9999),
        EventRange(EventType.SET_SPEED_MULT, 0, 80),
        EventRange(EventType.PAUSE, 0, 1024),
        EventRange(EventType.HOLD, 0, 100),
        # Converter emits 1 for present-or-not flags (not 0).
        EventRange(EventType.TWIRL, 0, 1),
        EventRange(EventType.MULTI_PLANET, 2, 10),
        EventRange(EventType.CHECKPOINT, 0, 1),
        EventRange(EventType.AUTO_PLAY_TILES, 0, 1),
        EventRange(EventType.SET_PLANET_ROTATION, 0, 10),
        EventRange(EventType.FREE_ROAM, 0, 1),
        EventRange(EventType.FREE_ROAM_TWIRL, 0, 1),
        EventRange(EventType.FREE_ROAM_REMOVE, 0, 1),
        EventRange(EventType.SCALE_MARGIN, 0, 400),
        EventRange(EventType.SCALE_RADIUS, 0, 400),
        EventRange(EventType.MULTITAP, 1, 10),
        EventRange(EventType.HIDE, 0, 1),
        EventRange(EventType.KILL_PLAYER, 0, 1),
        EventRange(EventType.POSITION_TRACK, 0, 1),
        EventRange(EventType.MOVE_TRACK, 0, 1),
        EventRange(EventType.COLOR_TRACK, 0, 10),
        EventRange(EventType.ANIMATE_TRACK, 0, 10),
        EventRange(EventType.MOVE_CAMERA, 0, 1),
        EventRange(EventType.CAMERA_POSITION_X, -1024, 1024),
        EventRange(EventType.CAMERA_POSITION_Y, -1024, 1024),
        EventRange(EventType.CAMERA_ROTATION, -8192, 8191),
        EventRange(EventType.CAMERA_ZOOM, 0, 2000),
        EventRange(EventType.CAMERA_DURATION, 0, 20480),
        EventRange(EventType.CAMERA_EASE, 0, 40),
        EventRange(EventType.CAMERA_RELATIVE, 0, 4),
        EventRange(EventType.SET_HITSOUND, 0, 10),
        EventRange(EventType.PLAY_SOUND, 0, 1),
        EventRange(EventType.SET_HOLD_SOUND, 0, 10),
        EventRange(EventType.REPEAT_EVENTS, 1, 1024),
        EventRange(EventType.SET_CONDITIONAL_EVENTS, 0, 1),
        EventRange(EventType.SET_INPUT_EVENT, 0, 1),
        EventRange(EventType.FLASH, 0, 4096),
        EventRange(EventType.BLOOM, 0, 4095),
        EventRange(EventType.SHAKE_SCREEN, 0, 512),
        EventRange(EventType.SET_FILTER, 0, 50),
        EventRange(EventType.SET_FILTER_ADVANCED, 0, 50),
        EventRange(EventType.FILTER_PROPERTIES, 0, 1),
        EventRange(EventType.BOOKMARK, 0, 1),
        EventRange(EventType.EDITOR_COMMENT, 0, 1),
        EventRange(EventType.CALL_METHOD, 0, 1),
        EventRange(EventType.ADD_COMPONENT, 0, 1),
        EventRange(EventType.CHANGE_TRACK, 0, 1),
        EventRange(EventType.FREE_ROAM_WARNING, 0, 1),
        EventRange(EventType.PAUSE_COUNTDOWN, 0, 20),
        EventRange(EventType.PAUSE_ANGLE_DIR, 0, 2),
        EventRange(EventType.HOLD_DISTANCE, 0, 400),
        EventRange(EventType.HOLD_LANDING, 0, 1),
        EventRange(EventType.TRACK_START_TILE, 0, 519),
        EventRange(EventType.TRACK_END_TILE, 0, 519),
        EventRange(EventType.VFX_PLANE, 0, 1),
        EventRange(EventType.VFX_COLOR, 0, 4095),
        EventRange(EventType.VFX_OPACITY, 0, 100),
        EventRange(EventType.VFX_ENABLED, 0, 1),
        EventRange(EventType.VFX_DISABLE_OTHERS, 0, 1),
        EventRange(EventType.VFX_INTENSITY, -10000, 65535),
        EventRange(EventType.VFX_STRENGTH, 0, 512),
        EventRange(EventType.VFX_THRESHOLD, 0, 100),
        EventRange(EventType.ANGLE_OFFSET, -16384, 16383),
    ]


def adofai_input_event_ranges() -> list[EventRange]:
    """Prefix/metadata types the converter also puts on the event stream."""
    return [
        EventRange(EventType.BPM, 1, 9999),
        EventRange(EventType.OFFSET, -32768, 32767),
        EventRange(EventType.PITCH, 50, 200),
    ]


def resolve_event_type(event_type, event_range: dict[EventType, EventRange]) -> EventType:
    if event_type in event_range:
        return event_type
    value = getattr(event_type, "value", event_type)
    try:
        canonical = EventType(value)
    except ValueError:
        return event_type
    if canonical in event_range:
        return canonical
    for key in event_range:
        if key.value == value:
            return key
    return event_type


def encode_event(
    event: Event,
    event_range: dict[EventType, EventRange],
    event_start: dict[EventType, int],
) -> int:
    """Same contract as ``Tokenizer.encode`` (unknown type / out-of-range)."""
    event_type = resolve_event_type(event.type, event_range)
    if event_type not in event_range:
        raise ValueError(f"unknown event type: {event.type}")

    er = event_range[event_type]
    offset = event_start[event_type]

    if not er.min_value <= event.value <= er.max_value:
        raise ValueError(
            f"event value {event.value} is not within range "
            f"[{er.min_value}, {er.max_value}] for event type {event.type}"
        )

    return offset + event.value - er.min_value


def _adofai_encode_tables(num_diff_classes: int = 24, max_time_shift: int = 4096):
    ranges = [
        EventRange(EventType.TIME_SHIFT, 0, max_time_shift),
        EventRange(EventType.SNAPPING, 0, 16),
        *adofai_event_ranges(),
        *adofai_input_event_ranges(),
        EventRange(EventType.DIFFICULTY, 0, num_diff_classes),
    ]
    event_range = {er.type: er for er in ranges}
    event_start: dict[EventType, int] = {}
    offset = 3
    for er in ranges:
        event_start[er.type] = offset
        offset += er.max_value - er.min_value + 1
    return event_range, event_start


def encode_adofai_events(events: list[Event], num_diff_classes: int = 24) -> list[int]:
    """Encode a converter event stream with the ADOFAI EventRange tables."""
    event_range, event_start = _adofai_encode_tables(num_diff_classes=num_diff_classes)
    return [encode_event(event, event_range, event_start) for event in events]
