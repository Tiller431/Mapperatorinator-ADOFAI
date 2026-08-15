"""
ADOFAI dataset for osuT5 training with lossless augmentation.

Implements:
- ROTATE: 8x multiplication by rotating all angles (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- Optional pitch shift (future)
"""

from __future__ import annotations

import random
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .data_utils import load_audio_file
from .adofai_parser import AdofaiParser
from ..tokenizer import Event, EventType, Tokenizer
from ..config import DataConfig

MILISECONDS_PER_SECOND = 1000
STEPS_PER_MILLISECOND = 0.1
LABEL_IGNORE_ID = -100


class AdofaiDataset(IterableDataset):
    """
    ADOFAI dataset with lossless augmentation.
    
    Dataset structure:
        <data_root>/
            <workshopId>__<chartName>/
                level.adofai
                audio.ogg (or .mp3, .wav, etc.)
    
    Augmentation:
        - ROTATE: 8 rotations (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
          Applied to all non-999 angles; actions stay on same floors
        - Optional pitch shift (disabled by default)
    
    N = 8 rotations × (1 + pitch_shifts) × base_charts
    """
    
    __slots__ = (
        "path",
        "start",
        "end",
        "args",
        "parser",
        "tokenizer",
        "chart_dirs",
        "test",
        "shared",
        "rotation_angles",
        "apply_rotation",
    )
    
    def __init__(
        self,
        args: DataConfig,
        parser: AdofaiParser,
        tokenizer: Tokenizer,
        chart_dirs: Optional[list[Path]] = None,
        test: bool = False,
        shared: Namespace = None,
    ):
        """
        Initialize ADOFAI dataset.
        
        Args:
            args: Data loading arguments
            parser: ADOFAI parser instance
            tokenizer: osuT5 tokenizer with ADOFAI events
            chart_dirs: Optional list of chart directories to use
            test: Whether this is test dataset
            shared: Shared namespace for progress tracking
        """
        super().__init__()
        self.path = Path(args.test_dataset_path if test else args.train_dataset_path)
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        
        # Lossless rotation augmentation
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 rotations
        self.apply_rotation = getattr(args, 'adofai_rotate_augment', True)
        
        # Find all chart directories
        if chart_dirs is not None:
            self.chart_dirs = chart_dirs
        else:
            self.chart_dirs = self._find_chart_dirs()
    
    def _find_chart_dirs(self) -> list[Path]:
        """
        Find all chart directories in the dataset path.
        
        Returns:
            List of paths to chart directories (each contains level.adofai + audio)
        """
        chart_dirs = []
        
        if not self.path.exists():
            print(f"Warning: Dataset path does not exist: {self.path}")
            return chart_dirs
        
        # Find all directories containing level.adofai
        for item in self.path.iterdir():
            if item.is_dir():
                adofai_file = item / "level.adofai"
                if adofai_file.exists():
                    chart_dirs.append(item)
        
        chart_dirs = sorted(chart_dirs)
        
        # Apply start/end range
        if self.end > 0:
            chart_dirs = chart_dirs[self.start:self.end]
        else:
            chart_dirs = chart_dirs[self.start:]
        
        print(f"Found {len(chart_dirs)} ADOFAI charts in {self.path}")
        if self.apply_rotation:
            print(f"With {len(self.rotation_angles)}x rotation augmentation = {len(chart_dirs) * len(self.rotation_angles)} variants")
        
        return chart_dirs
    
    def _rotate_events(self, events: list[Event], rotation_degrees: int) -> list[Event]:
        """
        Apply lossless rotation to angle events.
        
        Rotates all TILE_ANGLE events by rotation_degrees.
        Midspin (999) stays unchanged.
        All other events (actions, timing, etc.) remain on the same floors.
        
        Args:
            events: Original event list
            rotation_degrees: Degrees to rotate (0, 45, 90, 135, 180, 225, 270, 315)
            
        Returns:
            Rotated event list
        """
        if rotation_degrees == 0:
            return events
        
        rotated = []
        for event in events:
            if event.type == EventType.TILE_ANGLE:
                # Rotate angle
                new_angle = (event.value + rotation_degrees) % 360
                rotated.append(Event(EventType.TILE_ANGLE, new_angle))
            else:
                # Keep all other events unchanged (including MIDSPIN, actions, timing, etc.)
                rotated.append(event)
        
        return rotated
    
    def __iter__(self):
        """
        Iterate over dataset with augmentation.
        
        Yields samples with structure matching OsuParser output.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-worker: split charts among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            charts = [c for i, c in enumerate(self.chart_dirs) if i % num_workers == worker_id]
        else:
            charts = self.chart_dirs
        
        # Shuffle for training
        if not self.test:
            random.shuffle(charts)
        
        for chart_dir in charts:
            adofai_path = chart_dir / "level.adofai"
            
            try:
                # Parse chart
                events, audio_path = self.parser.parse(adofai_path)
                
                if audio_path is None or not audio_path.exists():
                    print(f"Warning: No audio found for {chart_dir.name}")
                    continue
                
                # Load audio
                audio = load_audio_file(
                    audio_path,
                    sample_rate=self.args.sample_rate if hasattr(self.args, 'sample_rate') else 16000
                )
                
                if audio is None:
                    print(f"Warning: Failed to load audio {audio_path}")
                    continue
                
                # Generate augmented variants
                rotations = self.rotation_angles if self.apply_rotation else [0]
                
                for rotation in rotations:
                    # Apply rotation augmentation
                    augmented_events = self._rotate_events(events, rotation)
                    
                    # TODO: Process events through tokenizer and create training sample
                    # For now, yield raw events + audio (will be processed by dataloader)
                    yield {
                        'events': augmented_events,
                        'audio': audio,
                        'chart_name': chart_dir.name,
                        'rotation': rotation,
                    }
            
            except Exception as e:
                print(f"Error processing {chart_dir.name}: {e}")
                continue
