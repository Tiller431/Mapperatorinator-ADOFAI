"""Pure lossless ADOFAI augmentations (no torch). Applied to raw angleData + actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ADOFAI_DIFFICULTY_PROXY = 5.0


def rotate_xy(x: float, y: float, rotate_deg: float) -> list[float]:
    rad = np.deg2rad(rotate_deg)
    cos_r = float(np.cos(rad))
    sin_r = float(np.sin(rad))
    return [x * cos_r - y * sin_r, x * sin_r + y * cos_r]


def apply_rotation(angle_data: list, actions: list[dict], rotate_deg: float) -> tuple[list, list[dict]]:
    rotated_angles = [(int(a + rotate_deg) % 360) if a != 999 else 999 for a in angle_data]
    rotated_actions = []
    for action in actions:
        act = dict(action)
        event_type = act.get("eventType", "")
        if event_type in ("MoveCamera", "PositionTrack", "AnimateTrack"):
            pos = act.get("position", [0, 0])
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                act["position"] = rotate_xy(float(pos[0]), float(pos[1]), rotate_deg)
            if "rotation" in act:
                act["rotation"] = (float(act["rotation"]) + rotate_deg) % 360
        elif event_type == "MoveTrack":
            pos_offset = act.get("positionOffset", [0, 0])
            if isinstance(pos_offset, (list, tuple)) and len(pos_offset) == 2:
                act["positionOffset"] = rotate_xy(float(pos_offset[0]), float(pos_offset[1]), rotate_deg)
        rotated_actions.append(act)
    return rotated_angles, rotated_actions


def apply_reflection(angle_data: list, actions: list[dict], reflect_fn) -> tuple[list, list[dict]]:
    reflected_angles = [reflect_fn(a) for a in angle_data]
    has_floor_0_twirl = any(
        act.get("floor") == 0 and act.get("eventType") == "Twirl" for act in actions
    )
    if not has_floor_0_twirl:
        return reflected_angles, [{"floor": 0, "eventType": "Twirl"}] + list(actions)
    return reflected_angles, [
        act for act in actions
        if not (act.get("floor") == 0 and act.get("eventType") == "Twirl")
    ]


def apply_matched_rate(settings: dict, actions: list[dict], rate_factor: float) -> tuple[dict, list[dict]]:
    if rate_factor == 1.0:
        return settings, actions
    transformed_settings = dict(settings)
    if "bpm" in transformed_settings:
        transformed_settings["bpm"] = float(transformed_settings["bpm"]) * rate_factor
    if "offset" in transformed_settings:
        transformed_settings["offset"] = float(transformed_settings["offset"]) / rate_factor
    transformed_actions = []
    for action in actions:
        act = dict(action)
        if act.get("eventType") == "SetSpeed" and act.get("speedType") == "Bpm":
            act["beatsPerMinute"] = float(act.get("beatsPerMinute", 0)) * rate_factor
        transformed_actions.append(act)
    return transformed_settings, transformed_actions


def resolve_difficulty(chart_dir: Path, settings: dict, default: float = ADOFAI_DIFFICULTY_PROXY) -> float:
    raw = settings.get("difficulty")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    for name in ("index.json", "metadata.json"):
        path = chart_dir / name
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("difficulty", payload.get("Difficulty"))
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
    parent_index = chart_dir.parent / "index.json"
    if parent_index.exists():
        try:
            payload = json.loads(parent_index.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            entry = payload.get(chart_dir.name, payload.get("charts", {}).get(chart_dir.name) if isinstance(payload.get("charts"), dict) else None)
            if isinstance(entry, dict):
                value = entry.get("difficulty", entry.get("Difficulty"))
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
    return float(default)
