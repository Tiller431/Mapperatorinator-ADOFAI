"""ADOFAI EventRange tables and encode used by Tokenizer and smoke tests.

Kept torch-free so tests can encode converter output without importing tokenizer.py.

Wide signed fields (ANGLE_OFFSET, VFX_INTENSITY) use a shared signed-hybrid
bucket map: 1-wide bins near zero, geometric bins out to the 126-set extrema
plus headroom. Tokenizer.encode/decode and CPU encode/decode must use this
map so train ``Embedding(vocab_size_out, 384)`` stays small while every event
still encodes. Decode reconstructs a legal integer (not bit-exact for the
log region). TIME_SHIFT stays a raw range plus chunking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .event import Event, EventRange, EventType


@dataclass(frozen=True)
class SignedHybridBuckets:
    """Signed buckets: exact ints in ``[-linear_abs, linear_abs]``, log beyond.

    Index layout (``min_value=0``):
    negative log (most negative first), negative linear, 0, positive linear,
    positive log. Values outside ``[-max_abs, max_abs]`` clamp to the edge
    bucket so encode never drops the event.
    """

    linear_abs: int
    max_abs: int
    log_subdiv: int = 8

    @property
    def n_log(self) -> int:
        if self.max_abs <= self.linear_abs:
            return 0
        return max(1, math.ceil(math.log2(self.max_abs / self.linear_abs) * self.log_subdiv))

    @property
    def n_bins(self) -> int:
        return 1 + 2 * self.linear_abs + 2 * self.n_log

    @property
    def zero_idx(self) -> int:
        return self.n_log + self.linear_abs

    def _log_edge(self, i: int) -> float:
        if self.n_log == 0:
            return float(self.linear_abs)
        ratio = self.max_abs / self.linear_abs
        return float(self.linear_abs) * (ratio ** (i / self.n_log))

    def quantize(self, value: int) -> int:
        value = int(value)
        if value == 0:
            return self.zero_idx
        sign = 1 if value > 0 else -1
        mag = min(abs(value), self.max_abs)
        if mag <= self.linear_abs:
            return self.zero_idx + sign * mag
        lo, hi = 1, self.n_log
        while lo < hi:
            mid = (lo + hi) // 2
            if mag <= self._log_edge(mid):
                hi = mid
            else:
                lo = mid + 1
        log_idx = lo - 1
        if sign > 0:
            return self.zero_idx + self.linear_abs + 1 + log_idx
        return self.n_log - 1 - log_idx

    def dequantize(self, bucket: int) -> int:
        bucket = int(bucket)
        if bucket < 0:
            bucket = 0
        elif bucket >= self.n_bins:
            bucket = self.n_bins - 1
        if bucket == self.zero_idx:
            return 0
        if bucket < self.n_log:
            log_idx = self.n_log - 1 - bucket
            lo = self._log_edge(log_idx)
            hi = self._log_edge(log_idx + 1)
            return -int(round(math.sqrt(lo * hi)))
        if bucket < self.zero_idx:
            return -(self.zero_idx - bucket)
        pos = bucket - self.zero_idx
        if pos <= self.linear_abs:
            return pos
        log_idx = pos - self.linear_abs - 1
        lo = self._log_edge(log_idx)
        hi = self._log_edge(log_idx + 1)
        return int(round(math.sqrt(lo * hi)))


# 126-set extrema: ANGLE_OFFSET ≈ -1e8..17280, VFX_INTENSITY ±1e6. Headroom
# covers the previous raw table edges (±2^27 / ±2^20) without 1-wide bins.
ANGLE_OFFSET_BUCKETS = SignedHybridBuckets(linear_abs=2048, max_abs=150_000_000, log_subdiv=8)
VFX_INTENSITY_BUCKETS = SignedHybridBuckets(linear_abs=256, max_abs=1_500_000, log_subdiv=8)

BUCKET_SPECS: dict[EventType, SignedHybridBuckets] = {
    EventType.ANGLE_OFFSET: ANGLE_OFFSET_BUCKETS,
    EventType.VFX_INTENSITY: VFX_INTENSITY_BUCKETS,
}


def quantize_adofai_value(event_type: EventType, value: int) -> int:
    spec = BUCKET_SPECS.get(event_type)
    if spec is None:
        return int(value)
    return spec.quantize(value)


def dequantize_adofai_value(event_type: EventType, stored: int) -> int:
    spec = BUCKET_SPECS.get(event_type)
    if spec is None:
        return int(stored)
    return spec.dequantize(stored)


def adofai_event_ranges() -> list[EventRange]:
    """Ranges for types ``AdofaiConverter.level_to_events`` emits on the map stream.

    Camera/VFX/gameplay bounds are widened from Tiller727/adofai-charts-v1
    (Workshop ``level.adofai``) plus known smoke overflows (zoom 1000,
    RepeatEvents 48, filter intensity 500, speed mult 1280, bloom -1752,
    shake 4333). ANGLE_OFFSET and VFX_INTENSITY are signed-hybrid bucket
    indices (not 2^28 / 2^21 raw bins); converter still emits raw ints and
    encode/decode apply ``BUCKET_SPECS``. Other values stay 1-wide.
    """
    return [
        EventRange(EventType.TILE_ANGLE, 0, 359),
        EventRange(EventType.MIDSPIN, 0, 0),
        EventRange(EventType.SET_SPEED_BPM, 1, 9999),
        EventRange(EventType.SET_SPEED_MULT, 0, 2047),
        EventRange(EventType.PAUSE, 0, 1024),
        EventRange(EventType.HOLD, 0, 100),
        # Converter emits 1 for present-or-not flags (not 0).
        EventRange(EventType.TWIRL, 0, 1),
        EventRange(EventType.MULTI_PLANET, 2, 10),
        EventRange(EventType.CHECKPOINT, 0, 1),
        EventRange(EventType.AUTO_PLAY_TILES, 0, 1),
        EventRange(EventType.SET_PLANET_ROTATION, 0, 31),
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
        EventRange(EventType.BLOOM, -4096, 4095),
        EventRange(EventType.SHAKE_SCREEN, 0, 8191),
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
        EventRange(EventType.VFX_INTENSITY, 0, VFX_INTENSITY_BUCKETS.n_bins - 1),
        EventRange(EventType.VFX_STRENGTH, 0, 2047),
        EventRange(EventType.VFX_THRESHOLD, 0, 100),
        EventRange(EventType.ANGLE_OFFSET, 0, ANGLE_OFFSET_BUCKETS.n_bins - 1),
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
    value = quantize_adofai_value(event_type, event.value)

    if not er.min_value <= value <= er.max_value:
        raise ValueError(
            f"event value {event.value} is not within range "
            f"[{er.min_value}, {er.max_value}] for event type {event.type}"
        )

    return offset + value - er.min_value


def decode_event(
    token_id: int,
    ranges: list[EventRange],
    event_start: dict[EventType, int],
) -> Event:
    """Inverse of ``encode_event``; bucketed types reconstruct a legal int."""
    for er in ranges:
        start = event_start[er.type]
        width = er.max_value - er.min_value
        if start <= token_id <= start + width:
            stored = er.min_value + token_id - start
            return Event(type=er.type, value=dequantize_adofai_value(er.type, stored))
    raise ValueError(f"id {token_id} is not mapped to any event")


def _adofai_encode_tables(num_diff_classes: int = 24, max_time_shift: int = 1_048_575):
    # Converter emits absolute ms. 4096 overflowed real Workshop charts
    # (first fail 4437; 126-chart max 921354). 2^20-1 plus chunking means
    # TIME_SHIFT is never dropped. Training Tokenizer still computes its
    # own windowed range from sequence length.
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
    return ranges, event_range, event_start


def _encode_time_shift_chunks(
    value: int,
    event_range: dict[EventType, EventRange],
    event_start: dict[EventType, int],
) -> list[int]:
    """Encode one TIME_SHIFT, splitting values above the table max (no drop)."""
    er = event_range[EventType.TIME_SHIFT]
    value = int(value)
    if value < er.min_value:
        raise ValueError(
            f"event value {value} is not within range "
            f"[{er.min_value}, {er.max_value}] for event type {EventType.TIME_SHIFT}"
        )
    tokens = []
    while value > er.max_value:
        tokens.append(encode_event(Event(EventType.TIME_SHIFT, er.max_value), event_range, event_start))
        value -= er.max_value
    tokens.append(encode_event(Event(EventType.TIME_SHIFT, value), event_range, event_start))
    return tokens


def encode_adofai_events(events: list[Event], num_diff_classes: int = 24) -> list[int]:
    """Encode a converter event stream with the ADOFAI EventRange tables."""
    _ranges, event_range, event_start = _adofai_encode_tables(num_diff_classes=num_diff_classes)
    tokens: list[int] = []
    for event in events:
        event_type = resolve_event_type(event.type, event_range)
        if event_type == EventType.TIME_SHIFT:
            tokens.extend(_encode_time_shift_chunks(event.value, event_range, event_start))
        else:
            tokens.append(encode_event(event, event_range, event_start))
    return tokens


def decode_adofai_events(token_ids: list[int], num_diff_classes: int = 24) -> list[Event]:
    """Decode CPU encode tokens; bucketed fields become reconstructed ints."""
    ranges, _event_range, event_start = _adofai_encode_tables(num_diff_classes=num_diff_classes)
    return [decode_event(int(token_id), ranges, event_start) for token_id in token_ids]
