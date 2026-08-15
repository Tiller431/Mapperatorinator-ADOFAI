"""Converter-emitted ADOFAI events must encode without ValueError.

Smoke on the pod died at ``Tokenizer.encode``:
``ValueError: unknown event type: EventType.MOVE_CAMERA``.
The converter emits MOVE_CAMERA (value 1) plus VFX; EventRange must cover those.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adofai.converter import AdofaiConverter
from adofai.parser import AdofaiLevel
from osuT5.osuT5.adofai_vocab import encode_adofai_events
from osuT5.osuT5.event import EventType


def _must_have_level() -> AdofaiLevel:
    """Short chart with MoveCamera, Flash, and the other must-have actions."""
    return AdofaiLevel(
        settings={"bpm": 120, "offset": 0, "pitch": 100, "difficulty": 8},
        angle_data=[0, 90, 999, 180],
        actions=[
            {"floor": 0, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 140, "angleOffset": 0},
            {"floor": 0, "eventType": "Twirl"},
            {"floor": 0, "eventType": "Pause", "duration": 1.0, "countdownTicks": 0, "angleCorrectionDir": -1},
            {"floor": 0, "eventType": "Hold", "duration": 1.0, "distanceMultiplier": 100, "landingAnimation": False},
            {"floor": 0, "eventType": "MultiPlanet", "planets": "ThreePlanets"},
            {
                "floor": 1,
                "eventType": "MoveCamera",
                "position": [4, -2],
                "rotation": 45,
                "zoom": 120,
                "duration": 2.0,
                "ease": "OutQuad",
                "relativeTo": "LastPositionNoRotation",
                "angleOffset": 0,
            },
            {
                "floor": 1,
                "eventType": "MoveTrack",
                "positionOffset": [1, 2],
                "startTile": [0, "ThisTile"],
                "endTile": [0, "ThisTile"],
                "duration": 1.0,
                "ease": "Linear",
                "angleOffset": 0,
            },
            {"floor": 1, "eventType": "PositionTrack", "position": [3, 1]},
            {"floor": 1, "eventType": "ColorTrack", "trackColorType": "Single"},
            {"floor": 1, "eventType": "AnimateTrack", "trackAnimation": "None"},
            {"floor": 1, "eventType": "AutoPlayTiles", "enabled": True},
            {"floor": 1, "eventType": "SetPlanetRotation", "easeParts": 1},
            {"floor": 1, "eventType": "FreeRoam"},
            {"floor": 1, "eventType": "FreeRoamTwirl"},
            {"floor": 1, "eventType": "FreeRoamRemove"},
            {"floor": 1, "eventType": "ScaleMargin", "scale": 100},
            {"floor": 1, "eventType": "ScaleRadius", "scale": 100},
            {"floor": 1, "eventType": "Multitap", "presses": 2},
            {"floor": 1, "eventType": "Hide", "hideJudgment": False},
            {"floor": 1, "eventType": "RepeatEvents", "repetitions": 1},
            {"floor": 1, "eventType": "SetConditionalEvents"},
            {"floor": 1, "eventType": "SetInputEvent"},
            {"floor": 1, "eventType": "SetHitsound", "hitsound": "Kick"},
            {"floor": 1, "eventType": "PlaySound"},
            {"floor": 1, "eventType": "SetHoldSound", "hitsound": "Kick"},
            {
                "floor": 1,
                "eventType": "Flash",
                "duration": 1.5,
                "plane": "Foreground",
                "startColor": "ffffff",
                "startOpacity": 100,
                "endColor": "000000",
                "endOpacity": 0,
                "angleOffset": 0,
                "ease": "Linear",
            },
            {
                "floor": 1,
                "eventType": "Bloom",
                "enabled": "Enabled",
                "intensity": 90,
                "threshold": 40,
                "color": "ffffff",
            },
            {"floor": 1, "eventType": "ShakeScreen", "intensity": 80, "strength": 50},
            {
                "floor": 1,
                "eventType": "SetFilter",
                "filter": "Grayscale",
                "enabled": "Enabled",
                "intensity": 70,
                "disableOthers": False,
            },
            {"floor": 1, "eventType": "Checkpoint"},
            {"floor": 1, "eventType": "KillPlayer"},
        ],
        decorations=[],
    )


REQUIRED_TYPES = {
    EventType.SET_SPEED_BPM,
    EventType.TWIRL,
    EventType.PAUSE,
    EventType.HOLD,
    EventType.MULTI_PLANET,
    EventType.MOVE_CAMERA,
    EventType.CAMERA_POSITION_X,
    EventType.CAMERA_POSITION_Y,
    EventType.CAMERA_ROTATION,
    EventType.CAMERA_ZOOM,
    EventType.CAMERA_DURATION,
    EventType.CAMERA_EASE,
    EventType.CAMERA_RELATIVE,
    EventType.MOVE_TRACK,
    EventType.POSITION_TRACK,
    EventType.COLOR_TRACK,
    EventType.ANIMATE_TRACK,
    EventType.AUTO_PLAY_TILES,
    EventType.SET_PLANET_ROTATION,
    EventType.FREE_ROAM,
    EventType.FREE_ROAM_TWIRL,
    EventType.FREE_ROAM_REMOVE,
    EventType.SCALE_MARGIN,
    EventType.SCALE_RADIUS,
    EventType.MULTITAP,
    EventType.HIDE,
    EventType.REPEAT_EVENTS,
    EventType.SET_CONDITIONAL_EVENTS,
    EventType.SET_INPUT_EVENT,
    EventType.SET_HITSOUND,
    EventType.PLAY_SOUND,
    EventType.SET_HOLD_SOUND,
    EventType.FLASH,
    EventType.BLOOM,
    EventType.SHAKE_SCREEN,
    EventType.SET_FILTER,
    EventType.CHECKPOINT,
    EventType.KILL_PLAYER,
    EventType.TILE_ANGLE,
    EventType.MIDSPIN,
    EventType.BPM,
    EventType.OFFSET,
    EventType.DIFFICULTY,
    EventType.TIME_SHIFT,
}


def test_converter_emits_move_camera_and_flash():
    events, _ = AdofaiConverter().level_to_events(_must_have_level())
    types = {event.type for event in events}
    assert EventType.MOVE_CAMERA in types
    assert EventType.FLASH in types
    camera = next(event for event in events if event.type == EventType.MOVE_CAMERA)
    assert camera.value == 1


def test_encode_accepts_event_type_from_other_module_identity():
    """train.py vs inference.py can bind two EventType classes with the same .value."""
    from osuT5.osuT5.adofai_vocab import encode_adofai_events
    from osuT5.osuT5.event import Event

    class OtherMoveCamera:
        value = EventType.MOVE_CAMERA.value

    event = Event(EventType.MOVE_CAMERA, 1)
    event.type = OtherMoveCamera()  # type: ignore[assignment]
    token_ids = encode_adofai_events([event])
    assert len(token_ids) == 1


def test_encode_move_camera_and_vfx_chart_without_valueerror():
    events, _ = AdofaiConverter().level_to_events(_must_have_level())
    types = {event.type for event in events}
    missing = REQUIRED_TYPES - types
    assert not missing, f"converter did not emit {sorted(t.value for t in missing)}"

    token_ids = encode_adofai_events(events)
    assert len(token_ids) == len(events)
    assert all(isinstance(token_id, int) for token_id in token_ids)


if __name__ == "__main__":
    test_converter_emits_move_camera_and_flash()
    print("converter emits MOVE_CAMERA+FLASH: ok")
    test_encode_move_camera_and_vfx_chart_without_valueerror()
    print("encode MOVE_CAMERA+VFX chart: ok")
