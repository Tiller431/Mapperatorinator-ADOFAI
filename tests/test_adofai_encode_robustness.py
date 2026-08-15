"""CPU encode buckets: messy Workshop JSON, SharpFAI null/enums, long TIME_SHIFT.

Real Tiller727/adofai-charts-v1 failures on main:
- parse: 2346220412__main missing comma / JSON5 quirks
- convert: 2980908404__main int(None) on MoveCamera.position
- convert: int('Backward') on Pause.angleCorrectionDir (SharpFAI enum)
- encode: 2118291532__main TIME_SHIFT 4437 (and later 20082+)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adofai.converter import AdofaiConverter
from adofai.parser import AdofaiLevel, parse_adofai, _clean_json_string
from osuT5.osuT5.adofai_vocab import encode_adofai_events
from osuT5.osuT5.event import Event, EventType


def _parse_via_json_cleanup(path: Path):
    """Official no-json5 path: cleanup must be valid for json.loads."""
    content = path.read_text(encoding="utf-8-sig")
    return json.loads(_clean_json_string(content))


def _write_messy_fixture(tmp_path: Path) -> Path:
    # Raw control char (BEL) inside a string — Workshop files do this.
    content = (
        "{\n"
        '\tsettings: {\n'
        '\t\tbpm: 120,\n'
        '\t\toffset: 0,\n'
        '\t\tpitch: 100,\n'
        '\t\tsongFilename: "test.ogg",\n'
        '\t\tlevelDesc: "line1\x07line2\nline3",\n'
        '\t\tdifficulty: 1, ,\n'
        "\t},\n"
        '\t"angleData": [0, 90, 180,],\n'
        '\t"actions": [\n'
        '\t\t{ "floor": 0, "eventType": "Twirl" }\n'
        '\t\t{ "floor": 1, "eventType": "Twirl" },\n'
        "\t],\n"
        '\t"decorations": [],\n'
        "}\n"
    )
    path = tmp_path / "messy.adofai"
    path.write_text(content, encoding="utf-8")
    return path


def test_messy_json_fixture_parses(tmp_path):
    """Trailing comma, unquoted key, control char, missing comma between objects."""
    path = _write_messy_fixture(tmp_path)
    level = parse_adofai(path)
    assert level.settings["bpm"] == 120
    assert level.angle_data == [0, 90, 180]
    assert len(level.actions) == 2
    assert level.actions[0]["eventType"] == "Twirl"
    assert level.actions[1]["floor"] == 1


def _write_settings_fixture(tmp_path: Path, name: str, extra_settings: str) -> Path:
    content = (
        "{\n"
        '\t"angleData": [0, 90],\n'
        '\t"settings": {\n'
        '\t\t"version": 15,\n'
        '\t\t"bpm": 120,\n'
        '\t\toffset: 0,\n'
        '\t\t"pitch": 100,\n'
        f"{extra_settings}"
        "\t},\n"
        '\t"actions": [],\n'
        '\t"decorations": []\n'
        "}\n"
    )
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_artist_links_comma_separated_https_urls_parse(tmp_path):
    """Workshop artistLinks is one string of comma-joined https URLs.

    Cleanup must not treat `,https:` inside that string as an unquoted key
    (2723436598 / 2756222522 / 2784257948 / 2985588667 / 3048734750).
    """
    path = _write_settings_fixture(
        tmp_path,
        "artist_links_https.adofai",
        '\t\t"artistLinks": "https://youtu.be/cSYO23Yu1po , https://youtu.be/kr61UJ-NTSw",\n',
    )
    data = _parse_via_json_cleanup(path)
    links = data["settings"]["artistLinks"]
    assert "https://youtu.be/cSYO23Yu1po" in links
    assert "https://youtu.be/kr61UJ-NTSw" in links
    assert ':"' not in links
    assert links.count("https://") == 2
    level = parse_adofai(path)
    assert level.settings["artistLinks"] == links


def test_artist_links_missing_comma_between_strings_parse(tmp_path):
    """Adjacent string values in artistLinks with no comma."""
    path = _write_settings_fixture(
        tmp_path,
        "artist_links_adjacent.adofai",
        '\t\t"artistLinks": ["https://evinak.bandcamp.com/" "https://soundcloud.com/sakura3zuki"],\n',
    )
    data = _parse_via_json_cleanup(path)
    links = data["settings"]["artistLinks"]
    assert links == [
        "https://evinak.bandcamp.com/",
        "https://soundcloud.com/sakura3zuki",
    ]
    level = parse_adofai(path)
    assert level.settings["artistLinks"] == links


def test_level_desc_raw_newline_and_tab_in_string_parse(tmp_path):
    """Raw CR/LF/TAB inside levelDesc (3469661239__main). Python json strict."""
    desc_body = (
        "spring map!\r\n"
        "A collab map with a spring feeling.\t\r\n"
        "Special Thanks : Pharah"
    )
    path = _write_settings_fixture(
        tmp_path,
        "level_desc_raw_breaks.adofai",
        f'\t\t"levelDesc": "{desc_body}",\n',
    )
    data = _parse_via_json_cleanup(path)
    desc = data["settings"]["levelDesc"]
    assert "spring map!" in desc
    assert "A collab map with a spring feeling." in desc
    assert "Special Thanks : Pharah" in desc
    level = parse_adofai(path)
    assert "spring map!" in level.settings["levelDesc"]


def test_int_none_camera_fields_do_not_raise():
    """MoveCamera position/rotation/zoom/relativeTo can be null on disk."""
    level = AdofaiLevel(
        settings={"bpm": 120, "offset": 0, "pitch": 100},
        angle_data=[0, 90],
        actions=[
            {
                "floor": 0,
                "eventType": "MoveCamera",
                "position": [None, None],
                "rotation": None,
                "zoom": None,
                "duration": 1.0,
                "ease": "Linear",
                "relativeTo": None,
                "angleOffset": None,
            }
        ],
        decorations=[],
    )
    events, times = AdofaiConverter().level_to_events(level)
    assert any(e.type == EventType.MOVE_CAMERA for e in events)
    pos_x = next(e for e in events if e.type == EventType.CAMERA_POSITION_X)
    pos_y = next(e for e in events if e.type == EventType.CAMERA_POSITION_Y)
    zoom = next(e for e in events if e.type == EventType.CAMERA_ZOOM)
    rot = next(e for e in events if e.type == EventType.CAMERA_ROTATION)
    assert pos_x.value == 0
    assert pos_y.value == 0
    assert zoom.value == 100
    assert rot.value == 0
    encode_adofai_events(events)


def test_int_backward_angle_correction_does_not_raise():
    """SharpFAI AngleCorrectionDirection.Backward = -1 (not a JS alias)."""
    level = AdofaiLevel(
        settings={"bpm": 120, "offset": 0, "pitch": 100},
        angle_data=[0, 90],
        actions=[
            {
                "floor": 0,
                "eventType": "Pause",
                "duration": 1.0,
                "countdownTicks": 0,
                "angleCorrectionDir": "Backward",
            }
        ],
        decorations=[],
    )
    events, _ = AdofaiConverter().level_to_events(level)
    pause_dir = next(e for e in events if e.type == EventType.PAUSE_ANGLE_DIR)
    # Backward = -1 → converter maps {-1: 0, 0: 1, 1: 2}
    assert pause_dir.value == 0
    encode_adofai_events(events)


def test_time_shift_4437_encodes():
    tokens = encode_adofai_events([Event(EventType.TIME_SHIFT, 4437)])
    assert tokens
    assert all(isinstance(t, int) for t in tokens)


def test_time_shift_20082_encodes():
    tokens = encode_adofai_events([Event(EventType.TIME_SHIFT, 20082)])
    assert tokens
    assert all(isinstance(t, int) for t in tokens)


def test_encode_leftover_126_set_overflows():
    """Values that failed encode after TIME_SHIFT was unblocked. Stored as-is."""
    extras = [
        Event(EventType.SET_SPEED_MULT, 160),
        Event(EventType.SET_SPEED_MULT, 240),
        Event(EventType.SET_SPEED_MULT, 1280),
        Event(EventType.VFX_INTENSITY, 77_777),
        Event(EventType.VFX_INTENSITY, 1_000_000),
        Event(EventType.VFX_INTENSITY, -1_000_000),
        Event(EventType.SHAKE_SCREEN, 800),
        Event(EventType.SHAKE_SCREEN, 4333),
        Event(EventType.BLOOM, -1752),
        Event(EventType.BLOOM, -27),
        Event(EventType.ANGLE_OFFSET, -114_514),
        Event(EventType.ANGLE_OFFSET, 17_280),
        Event(EventType.ANGLE_OFFSET, -100_000_000),
        Event(EventType.VFX_STRENGTH, 1000),
        Event(EventType.SET_PLANET_ROTATION, 14),
    ]
    tokens = encode_adofai_events(extras)
    assert len(tokens) == len(extras)
    assert all(isinstance(t, int) for t in tokens)


HUB_CHARTS = Path("/workspace/adofai-dataset/hub-package/charts")


def _hub_level(chart_dir: str) -> Path:
    path = HUB_CHARTS / chart_dir / "level.adofai"
    if not path.exists():
        pytest.skip(f"hub-package chart not mounted: {path}")
    return path


def test_hub_2346220412_parses_missing_comma():
    """Real first parse fail: Expecting ',' delimiter line 450."""
    path = _hub_level("2346220412__main")
    level = parse_adofai(path)
    assert level.angle_data
    events, _ = AdofaiConverter().level_to_events(level)
    tokens = encode_adofai_events(events)
    assert tokens


def test_hub_2980908404_converts_int_none():
    """Real first convert fail: int(None) on MoveCamera.position."""
    path = _hub_level("2980908404__main")
    level = parse_adofai(path)
    events, _ = AdofaiConverter().level_to_events(level)
    assert any(e.type == EventType.MOVE_CAMERA for e in events)
    encode_adofai_events(events)


def test_hub_2118291532_encodes_time_shift_4437():
    """Real first TIME_SHIFT overflow: 4437 (chart also has values past 20082)."""
    path = _hub_level("2118291532__main")
    level = parse_adofai(path)
    events, _ = AdofaiConverter().level_to_events(level)
    time_shifts = [e.value for e in events if e.type == EventType.TIME_SHIFT]
    assert any(v >= 4437 for v in time_shifts)
    assert any(v >= 20082 for v in time_shifts)
    tokens = encode_adofai_events(events)
    assert tokens


@pytest.mark.parametrize(
    "chart_dir",
    [
        "2723436598__main",
        "2756222522__main",
        "2784257948__main",
        "2985588667__main",
        "3048734750__main",
        "3469661239__main",
    ],
)
def test_hub_leftover_parse_adofai_six(chart_dir):
    """Six leftover official parse_adofai fails after PR #7 (artistLinks / levelDesc)."""
    path = _hub_level(chart_dir)
    level = parse_adofai(path)
    assert level.angle_data
    events, _ = AdofaiConverter().level_to_events(level)
    tokens = encode_adofai_events(events)
    assert tokens
