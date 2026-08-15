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


def _numeric_xy(pos) -> tuple[float, float] | None:
    """Return (x, y) only when both coords are finite numbers; else leave the field alone."""
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    x, y = pos[0], pos[1]
    if x is None or y is None:
        return None
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(xf) and np.isfinite(yf)):
        return None
    return xf, yf


def apply_rotation(angle_data: list, actions: list[dict], rotate_deg: float) -> tuple[list, list[dict]]:
    rotated_angles = [(int(a + rotate_deg) % 360) if a != 999 else 999 for a in angle_data]
    rotated_actions = []
    for action in actions:
        act = dict(action)
        event_type = act.get("eventType", "")
        if event_type in ("MoveCamera", "PositionTrack", "AnimateTrack"):
            xy = _numeric_xy(act.get("position"))
            if xy is not None:
                act["position"] = rotate_xy(xy[0], xy[1], rotate_deg)
            if "rotation" in act:
                act["rotation"] = (float(act["rotation"]) + rotate_deg) % 360
        elif event_type == "MoveTrack":
            xy = _numeric_xy(act.get("positionOffset"))
            if xy is not None:
                act["positionOffset"] = rotate_xy(xy[0], xy[1], rotate_deg)
        rotated_actions.append(act)
    return rotated_angles, rotated_actions


REFLECT_AXES = {
    # Angle formulas from the lock list; matching XY / camera rotation transforms.
    "x_flip": {
        "angle": lambda a: (-a) % 360 if a != 999 else 999,
        "xy": lambda x, y: [x, -y],
    },
    "y_flip": {
        "angle": lambda a: (180 - a) % 360 if a != 999 else 999,
        "xy": lambda x, y: [-x, y],
    },
    "diag_y_eq_x": {
        "angle": lambda a: (90 - a) % 360 if a != 999 else 999,
        "xy": lambda x, y: [y, x],
    },
    "diag_y_eq_neg_x": {
        "angle": lambda a: (270 - a) % 360 if a != 999 else 999,
        "xy": lambda x, y: [-y, -x],
    },
}


def apply_reflection(angle_data: list, actions: list[dict], axis: str = "x_flip") -> tuple[list, list[dict]]:
    spec = REFLECT_AXES[axis]
    reflect_fn = spec["angle"]
    reflect_xy = spec["xy"]
    reflected_angles = [reflect_fn(a) for a in angle_data]
    reflected_actions = []
    for action in actions:
        act = dict(action)
        event_type = act.get("eventType", "")
        if event_type in ("MoveCamera", "PositionTrack", "AnimateTrack"):
            xy = _numeric_xy(act.get("position"))
            if xy is not None:
                act["position"] = reflect_xy(xy[0], xy[1])
            if "rotation" in act:
                act["rotation"] = reflect_fn(float(act["rotation"]))
        elif event_type == "MoveTrack":
            xy = _numeric_xy(act.get("positionOffset"))
            if xy is not None:
                act["positionOffset"] = reflect_xy(xy[0], xy[1])
        reflected_actions.append(act)
    has_floor_0_twirl = any(
        act.get("floor") == 0 and act.get("eventType") == "Twirl" for act in reflected_actions
    )
    if not has_floor_0_twirl:
        return reflected_angles, [{"floor": 0, "eventType": "Twirl"}] + reflected_actions
    return reflected_angles, [
        act for act in reflected_actions
        if not (act.get("floor") == 0 and act.get("eventType") == "Twirl")
    ]


def _ola_time_stretch(samples: np.ndarray, target_len: int, win: int = 1024) -> np.ndarray:
    """Overlap-add time-stretch that keeps spectral pitch, restores duration."""
    x = np.asarray(samples, dtype=np.float64)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(x) == target_len:
        return x.astype(np.float32)
    win = int(min(win, len(x), target_len))
    if win < 8:
        t_src = np.linspace(0.0, 1.0, len(x), endpoint=False)
        t_dst = np.linspace(0.0, 1.0, target_len, endpoint=False)
        return np.interp(t_dst, t_src, x).astype(np.float32)
    hop_in = max(1, win // 4)
    n_frames = max(1, 1 + (len(x) - win) // hop_in)
    hop_out = max(1, int(round((target_len - win) / max(n_frames - 1, 1)))) if n_frames > 1 else 1
    window = np.hanning(win)
    out = np.zeros(max(target_len, (n_frames - 1) * hop_out + win) + 1, dtype=np.float64)
    norm = np.zeros_like(out)
    for i in range(n_frames):
        src_start = i * hop_in
        frame = np.zeros(win, dtype=np.float64)
        avail = max(0, min(win, len(x) - src_start))
        if avail:
            frame[:avail] = x[src_start:src_start + avail]
        dst_start = i * hop_out
        out[dst_start:dst_start + win] += frame * window
        norm[dst_start:dst_start + win] += window
    norm[norm < 1e-8] = 1.0
    return (out / norm)[:target_len].astype(np.float32)


def pitch_shift_same_duration(samples: np.ndarray, sample_rate: int, pitch: float) -> np.ndarray:
    """Same-duration pitch shift. Chart events stay untouched; settings.pitch is the knob.

    pitch=100 is identity. Other values resample (pitch+duration) then OLA-stretch
    back to the original length so duration is unchanged.
    """
    del sample_rate  # rate is unused; shift is defined by the game pitch percent
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0 or abs(float(pitch) - 100.0) < 1e-6:
        return samples
    ratio = float(pitch) / 100.0
    if ratio <= 0:
        return samples
    n = len(samples)
    n_pitched = max(1, int(round(n / ratio)))
    t_src = np.linspace(0.0, 1.0, n, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, n_pitched, endpoint=False)
    pitched = np.interp(t_dst, t_src, samples.astype(np.float64))
    return _ola_time_stretch(pitched, n)


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
