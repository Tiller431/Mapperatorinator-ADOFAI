"""
ADOFAI parser for osuT5 training.

Wraps adofai/parser.py and adofai/converter.py to provide OsuParser-compatible interface.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from osuT5.osuT5.event import Event
from osuT5.osuT5.config import TrainConfig
from osuT5.osuT5.tokenizer import Tokenizer
from adofai.parser import parse_adofai
from adofai.converter import AdofaiConverter


class AdofaiParser:
    """
    Parser for ADOFAI charts compatible with osuT5 training infrastructure.
    
    Provides similar interface to OsuParser but for ADOFAI .adofai files.
    """
    
    def __init__(self, args: TrainConfig, tokenizer: Tokenizer):
        """
        Initialize ADOFAI parser.
        
        Args:
            args: Training configuration
            tokenizer: osuT5 tokenizer with ADOFAI event ranges
        """
        self.args = args
        self.tokenizer = tokenizer
        self.converter = AdofaiConverter()
    
    def parse(
        self,
        adofai_path: str | Path,
        audio_path: Optional[str | Path] = None,
    ) -> tuple[list[Event], Optional[Path]]:
        """
        Parse ADOFAI chart and return event sequence + audio path.
        
        Args:
            adofai_path: Path to .adofai file
            audio_path: Optional explicit audio path (otherwise inferred from chart dir)
            
        Returns:
            Tuple of (events, audio_path) where:
                - events: List of osuT5 Event objects (timing, tiles, actions)
                - audio_path: Path to audio file or None if not found
        """
        adofai_path = Path(adofai_path)
        
        # Parse .adofai file
        level = parse_adofai(adofai_path)
        
        # Convert to osuT5 Events
        events, event_times = self.converter.level_to_events(level)
        
        # Find audio file if not provided
        if audio_path is None:
            chart_dir = adofai_path.parent
            audio_filename = level.settings.get("songFilename", "")
            
            # Try exact filename from settings
            if audio_filename:
                candidate = chart_dir / audio_filename
                if candidate.exists():
                    audio_path = candidate
            
            # Fall back to scanning for audio files
            if audio_path is None:
                audio_extensions = [".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac"]
                for ext in audio_extensions:
                    candidates = list(chart_dir.glob(f"*{ext}"))
                    if candidates:
                        audio_path = candidates[0]
                        break
        
        return events, Path(audio_path) if audio_path else None
