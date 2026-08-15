"""
ADOFAI parser for osuT5 training.

I/O only: wraps adofai/parser.py + adofai/converter.py and returns the same
`list[Event], list[int]` contract as OsuParser.parse / parse_timing.
Does not use adofai/tokenizer.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..event import Event, EventType
from ..config import TrainConfig
from ..tokenizer import Tokenizer
from adofai.parser import AdofaiLevel, parse_adofai
from adofai.converter import AdofaiConverter

TIMING_EVENT_TYPES = {
    EventType.TIME_SHIFT,
    EventType.BPM,
    EventType.OFFSET,
    EventType.SET_SPEED_BPM,
    EventType.SET_SPEED_MULT,
    EventType.PAUSE,
    EventType.HOLD,
}


class AdofaiParser:
    """OsuParser-compatible ADOFAI parser for osuT5 dataloaders."""

    def __init__(self, args: TrainConfig, tokenizer: Tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.converter = AdofaiConverter()

    def parse(
        self,
        level: AdofaiLevel | str | Path,
        speed: float = 1.0,
        song_length: Optional[float] = None,
    ) -> tuple[list[Event], list[int]]:
        """Parse a level into full map events (tiles, actions, camera, VFX)."""
        if not isinstance(level, AdofaiLevel):
            level = parse_adofai(level)
        events, event_times = self.converter.level_to_events(level)
        return events, [int(t) for t in event_times]

    def parse_timing(
        self,
        level: AdofaiLevel | str | Path,
        speed: float = 1.0,
        song_length: Optional[float] = None,
    ) -> tuple[list[Event], list[int]]:
        """Timing stage: BPM/offset/SetSpeed/Pause/Hold + TIME_SHIFT."""
        events, event_times = self.parse(level, speed, song_length)
        out_events = []
        out_times = []
        for event, time in zip(events, event_times):
            if event.type in TIMING_EVENT_TYPES:
                out_events.append(event)
                out_times.append(time)
        return out_events, out_times

    def find_audio_path(
        self,
        adofai_path: str | Path,
        level: Optional[AdofaiLevel] = None,
    ) -> Optional[Path]:
        adofai_path = Path(adofai_path)
        chart_dir = adofai_path.parent
        if level is None:
            level = parse_adofai(adofai_path)
        audio_filename = level.settings.get("songFilename", "")
        if audio_filename:
            candidate = chart_dir / audio_filename
            if candidate.exists():
                return candidate
        for ext in [".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac"]:
            candidates = list(chart_dir.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        return None
