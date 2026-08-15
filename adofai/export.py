"""
Export osuT5 Events to a parseable .adofai file.

Production generate path: repo-root Hydra `inference.py -cn adofai_v31`.
After Processor emits Events, this module calls `AdofaiConverter.events_to_level`
and writes SharpFAI on-disk keys. Decorations stay empty.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from osuT5.osuT5.event import Event, EventType

from .converter import AdofaiConverter
from .parser import AdofaiLevel, write_adofai


def events_to_adofai_file(
    events: list[Event],
    event_times: Optional[list[float]] = None,
    output_path: str | Path = None,
    base_settings: Optional[dict] = None,
) -> tuple[AdofaiLevel, Path]:
    """Convert Events to an AdofaiLevel and write a .adofai file.

    Midspin stays angleData 999. Empty angle sequences get a minimal 3-tile path
    from `events_to_level` so the file is always parseable.
    """
    if output_path is None:
        raise ValueError("output_path is required")
    if event_times is None:
        event_times = [0] * len(events)
    elif len(event_times) < len(events):
        event_times = list(event_times) + [event_times[-1] if event_times else 0] * (
            len(events) - len(event_times)
        )

    converter = AdofaiConverter()
    level = converter.events_to_level(events, event_times, base_settings)
    output_path = Path(output_path)
    write_adofai(level, output_path)
    return level, output_path


def adofai_output_path(output_dir: str | Path, title: Optional[str] = None) -> Path:
    """Pick a unique .adofai path under output_dir."""
    output_dir = Path(output_dir)
    raw = (title or "chart").strip() or "chart"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    return output_dir / f"{safe}_{uuid.uuid4().hex[:8]}.adofai"


def apply_inference_settings(
    level: AdofaiLevel,
    *,
    bpm: Optional[float] = None,
    offset: Optional[int] = None,
    difficulty: Optional[float] = None,
    events: Optional[list[Event]] = None,
) -> AdofaiLevel:
    """Fill settings from inference args when the event stream omitted them."""
    event_types = {event.type for event in (events or [])}
    if bpm is not None and EventType.BPM not in event_types:
        level.settings["bpm"] = bpm
    if offset is not None and EventType.OFFSET not in event_types:
        level.settings["offset"] = offset
    if difficulty is not None and EventType.DIFFICULTY not in event_types:
        level.settings["difficulty"] = difficulty
    return level


def tokens_to_events(tokenizer, token_ids: list[int]) -> list[Event]:
    """Decode tokenizer ids to Events, skipping PAD/SOS/EOS and unknown ids."""
    events = []
    skip = {tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id}
    for token_id in token_ids:
        if token_id in skip:
            continue
        try:
            events.append(tokenizer.decode(int(token_id)))
        except ValueError:
            continue
    return events
