"""Wide EventRanges must be bucketed so train Embedding(vocab, 384) fits in RAM.

Smoke on main died before a batch:

    RuntimeError: can't allocate 415673163264 bytes (~387 GiB)
    at nn.Embedding(config.vocab_size, d_model=384)

Cause: ANGLE_OFFSET [-2^27, 2^27) and VFX_INTENSITY [-2^20, 2^20) were stored
as 1-wide EventRange bins (vocab_size_out ≈ 270_620_549). CPU encode can keep
raw integers; the train table cannot inherit those widths.

Lossless exact-int is not required for these two fields. Encode must not drop
the event; decode/export must reconstruct a legal integer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adofai.converter import AdofaiConverter
from adofai.export import events_to_adofai_file
from adofai.parser import parse_adofai
from osuT5.osuT5.adofai_vocab import adofai_event_ranges, encode_adofai_events
from osuT5.osuT5.event import Event, EventType


# Well under the ~100k Embedding budget. 2^21 / 2^28 raw bins are forbidden.
_MAX_SINGLE_RANGE = 25_000
_MAX_VOCAB_OUT = 100_000
_MIN_BUCKETS = 100
_MAX_BUCKETS = 8_000

_ANGLE_EXTREMES = (-100_000_000, -114_514, 17_280, 134_217_727)
_VFX_EXTREMES = (-1_000_000, -1_048_576, 1_000_000, 1_048_575)

HUB_CHARTS = Path("/workspace/adofai-dataset/hub-package/charts")


def _range_by_type() -> dict[EventType, int]:
    return {er.type: er.max_value - er.min_value + 1 for er in adofai_event_ranges()}


def test_no_event_range_is_raw_2pow21_or_2pow28():
    """The two bug shapes: 1-wide bins over 2^21 / 2^28 explode Embedding."""
    widths = _range_by_type()
    assert widths[EventType.ANGLE_OFFSET] < (1 << 21), (
        f"ANGLE_OFFSET still has {widths[EventType.ANGLE_OFFSET]} raw bins"
    )
    assert widths[EventType.VFX_INTENSITY] < (1 << 20), (
        f"VFX_INTENSITY still has {widths[EventType.VFX_INTENSITY]} raw bins"
    )


def test_wide_fields_are_hundreds_to_low_thousands_of_bins():
    widths = _range_by_type()
    for event_type in (EventType.ANGLE_OFFSET, EventType.VFX_INTENSITY):
        width = widths[event_type]
        assert _MIN_BUCKETS <= width <= _MAX_BUCKETS, (
            f"{event_type} width {width} is not hundreds-to-low-thousands"
        )


def test_no_adofai_output_range_explodes_vocab():
    """Any leftover 1-wide range wide enough to blow vocab must also be bucketed."""
    for er in adofai_event_ranges():
        width = er.max_value - er.min_value + 1
        assert width <= _MAX_SINGLE_RANGE, (
            f"{er.type} width {width} would inflate vocab_size_out"
        )


def _adofai_v31_vocab_size_out() -> int:
    """Same output-table width Tokenizer builds for ``adofai_v31`` (torch-free).

    Mirrors ``Tokenizer.__init__``: offset 3 + SOS/EOS for none/timing/map,
    windowed TIME_SHIFT, SNAPPING, ``adofai_event_ranges()``, then the osu
    leftover types that are always appended. Does not import tokenizer.py
    (that module pulls torch via dataset.data_utils).
    """
    src_seq_len = 4096
    hop_length = 128
    sample_rate = 16000
    max_time_shift = int(
        ((src_seq_len - 1) * hop_length * 1000 / sample_rate) / 10
    )
    offset = 3 + 6  # PAD/SOS/EOS + 3 context types
    leftover_width = 1 + (2 ** 3 * 3 * 3 + 1) + 101 + 12  # combo/hitsound/vol/notes
    return (
        offset
        + (max_time_shift + 1)
        + 17
        + sum(er.max_value - er.min_value + 1 for er in adofai_event_ranges())
        + leftover_width
    )


def test_vocab_size_out_fits_embedding_384():
    """Train Tokenizer vocab must be small enough for Embedding(vocab, 384)."""
    vocab_size_out = _adofai_v31_vocab_size_out()
    assert vocab_size_out < _MAX_VOCAB_OUT, (
        f"vocab_size_out={vocab_size_out} still cannot fit Embedding(*, 384)"
    )
    # 270M * 384 * 4 bytes was ~387 GiB. Stay under ~150 MiB.
    embed_bytes = vocab_size_out * 384 * 4
    assert embed_bytes < 150 * 1024 * 1024


@pytest.mark.parametrize("value", _ANGLE_EXTREMES)
def test_extreme_angle_offset_still_encodes(value):
    tokens = encode_adofai_events([Event(EventType.ANGLE_OFFSET, value)])
    assert len(tokens) == 1
    assert isinstance(tokens[0], int)


@pytest.mark.parametrize("value", _VFX_EXTREMES)
def test_extreme_vfx_intensity_still_encodes(value):
    tokens = encode_adofai_events([Event(EventType.VFX_INTENSITY, value)])
    assert len(tokens) == 1
    assert isinstance(tokens[0], int)


@pytest.mark.parametrize(
    "event_type,value",
    [
        (EventType.ANGLE_OFFSET, -100_000_000),
        (EventType.ANGLE_OFFSET, 17_280),
        (EventType.ANGLE_OFFSET, -134_217_728),
        (EventType.VFX_INTENSITY, -1_000_000),
        (EventType.VFX_INTENSITY, 1_000_000),
        (EventType.VFX_INTENSITY, 1_048_575),
    ],
)
def test_roundtrip_does_not_drop_extreme_event(event_type, value):
    from osuT5.osuT5.adofai_vocab import decode_adofai_events

    source = Event(event_type, value)
    tokens = encode_adofai_events([source])
    decoded = decode_adofai_events(tokens)
    assert len(decoded) == 1
    assert decoded[0].type == event_type
    assert isinstance(decoded[0].value, int)
    # Reconstruct a legal signed int; exact lossless is not required.
    assert decoded[0].value != 0 or value == 0
    if value < 0:
        assert decoded[0].value < 0
    elif value > 0:
        assert decoded[0].value > 0


def test_small_values_roundtrip_exactly_for_playable_export():
    """Linear region must stay exact so typical charts export unchanged."""
    from osuT5.osuT5.adofai_vocab import decode_adofai_events

    source = [
        Event(EventType.ANGLE_OFFSET, 15),
        Event(EventType.ANGLE_OFFSET, -540),
        Event(EventType.VFX_INTENSITY, 70),
        Event(EventType.VFX_INTENSITY, -60),
    ]
    decoded = decode_adofai_events(encode_adofai_events(source))
    assert [event.type for event in decoded] == [event.type for event in source]
    assert [event.value for event in decoded] == [event.value for event in source]


def test_time_shift_chunking_not_regressed():
    from osuT5.osuT5.adofai_vocab import decode_adofai_events

    tokens = encode_adofai_events([Event(EventType.TIME_SHIFT, 921_354)])
    assert tokens
    decoded = decode_adofai_events(tokens)
    assert all(event.type == EventType.TIME_SHIFT for event in decoded)
    assert sum(event.value for event in decoded) == 921_354


def test_decoded_extreme_exports_playable_adofai(tmp_path):
    from osuT5.osuT5.adofai_vocab import decode_adofai_events

    events = [
        Event(EventType.BPM, 120),
        Event(EventType.TILE_ANGLE, 0),
        Event(EventType.TILE_ANGLE, 90),
        Event(EventType.SET_SPEED_BPM, 140),
        Event(EventType.ANGLE_OFFSET, -100_000_000),
        Event(EventType.SET_FILTER, 1),
        Event(EventType.VFX_INTENSITY, 1_000_000),
    ]
    decoded = decode_adofai_events(encode_adofai_events(events))
    assert any(event.type == EventType.ANGLE_OFFSET for event in decoded)
    assert any(event.type == EventType.VFX_INTENSITY for event in decoded)
    out = tmp_path / "bucketed.adofai"
    events_to_adofai_file(decoded, [0] * len(decoded), out, {"bpm": 120, "offset": 0})
    parsed = parse_adofai(out)
    assert parsed.angle_data
    speed = next(action for action in parsed.actions if action["eventType"] == "SetSpeed")
    assert isinstance(speed["angleOffset"], int)
    filt = next(action for action in parsed.actions if action["eventType"] == "SetFilter")
    assert isinstance(filt["intensity"], int)


def test_tokenizer_encode_decode_agrees_with_cpu_bucket_map():
    """Tokenizer.encode/decode and CPU encode/decode share BUCKET_SPECS."""
    from osuT5.osuT5.adofai_vocab import (
        decode_adofai_events,
        dequantize_adofai_value,
        encode_event,
        quantize_adofai_value,
    )

    raw = Event(EventType.ANGLE_OFFSET, -100_000_000)
    cpu_decoded = decode_adofai_events(encode_adofai_events([raw]))[0]
    stored = quantize_adofai_value(EventType.ANGLE_OFFSET, raw.value)
    assert dequantize_adofai_value(EventType.ANGLE_OFFSET, stored) == cpu_decoded.value

    raw_vfx = Event(EventType.VFX_INTENSITY, 1_000_000)
    assert dequantize_adofai_value(
        EventType.VFX_INTENSITY, quantize_adofai_value(EventType.VFX_INTENSITY, raw_vfx.value)
    ) == decode_adofai_events(encode_adofai_events([raw_vfx]))[0].value

    # Same token formula Tokenizer.encode uses (offset + quantized - min).
    event_range = {er.type: er for er in adofai_event_ranges()}
    event_start = {EventType.ANGLE_OFFSET: 100, EventType.VFX_INTENSITY: 200}
    token = encode_event(raw, event_range, event_start)
    er = event_range[EventType.ANGLE_OFFSET]
    stored_from_token = er.min_value + token - event_start[EventType.ANGLE_OFFSET]
    assert dequantize_adofai_value(EventType.ANGLE_OFFSET, stored_from_token) == cpu_decoded.value


def test_hub_all_126_charts_encode_without_skip():
    """Every hub-package chart must still encode. Skip only if the tree is absent."""
    if not HUB_CHARTS.is_dir():
        pytest.skip(f"hub-package charts not mounted: {HUB_CHARTS}")
    converter = AdofaiConverter()
    leftovers: list[str] = []
    chart_dirs = sorted(path for path in HUB_CHARTS.iterdir() if path.is_dir())
    assert chart_dirs, f"hub-package charts dir is empty: {HUB_CHARTS}"
    for chart_dir in chart_dirs:
        level_path = chart_dir / "level.adofai"
        if not level_path.exists():
            leftovers.append(f"{chart_dir.name}: missing level.adofai")
            continue
        try:
            from adofai.parser import parse_adofai as parse_level

            events, _ = converter.level_to_events(parse_level(level_path))
            tokens = encode_adofai_events(events)
            if not tokens:
                leftovers.append(f"{chart_dir.name}: empty token stream")
        except Exception as exc:  # noqa: BLE001 — collect leftovers, do not skip events
            leftovers.append(f"{chart_dir.name}: {type(exc).__name__}: {exc}")
    assert not leftovers, "hub-package encode leftovers:\n" + "\n".join(leftovers)
