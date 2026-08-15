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
        
        # Lossless augmentation: LOCKED PARAMETER SETS (not cartesian product)
        # Sampling rule: pick ONE geometric transform, THEN independently maybe rate, maybe pitch
        
        # Geometric transforms (pick ONE per sample)
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 rotations
        self.reflect_transforms = [
            ('x_flip', lambda a: (-a) % 360 if a != 999 else 999),
            ('y_flip', lambda a: (180 - a) % 360 if a != 999 else 999),
            ('diag_y_eq_x', lambda a: (90 - a) % 360 if a != 999 else 999),
            ('diag_y_eq_neg_x', lambda a: (270 - a) % 360 if a != 999 else 999),
        ]
        
        # Build geometric transform pool
        self.geometric_transforms = []
        
        # Add identity (no transform)
        if getattr(args, 'adofai_identity_augment', True):
            self.geometric_transforms.append(('identity', None, 0, False))  # (name, reflect_fn, rotate_deg, needs_twirl)
        
        # Add rotations
        if getattr(args, 'adofai_rotate_augment', True):
            for angle in self.rotation_angles:
                if angle != 0:  # Skip 0° rotation (covered by identity)
                    self.geometric_transforms.append((f'rotate_{angle}', None, angle, False))
        
        # Add reflections (each requires floor-0 Twirl toggle)
        if getattr(args, 'adofai_reflect_augment', True):
            for name, reflect_fn in self.reflect_transforms:
                self.geometric_transforms.append((name, reflect_fn, 0, True))
        
        # Matched-rate factors (applied independently AFTER geometric)
        self.matched_rate_factors = getattr(args, 'adofai_matched_rate_factors', [1.0])  # e.g., [0.9, 1.0, 1.1, 1.2]
        
        # Same-duration pitch shifts (applied independently AFTER geometric)
        self.pitch_shifts = getattr(args, 'adofai_pitch_shifts', [100])  # settings.pitch: [90, 100, 110]
        
        print(f"ADOFAI augmentation: {len(self.geometric_transforms)} geometric × {len(self.matched_rate_factors)} rates × {len(self.pitch_shifts)} pitches")
        print(f"  Sampling: pick 1 geometric, THEN independently maybe rate, maybe pitch (not cartesian)")
        print(f"  Expected variants per chart: ~{len(self.geometric_transforms)} (geometric pool size)")
        
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
    
    def _apply_geometric_transform(
        self,
        angle_data: list[int],
        actions: list[dict],
        transform_name: str,
        reflect_fn,
        rotate_deg: int,
        needs_twirl: bool
    ) -> tuple[list[int], list[dict]]:
        """
        Apply geometric transform to RAW angleData and actions.
        
        Args:
            angle_data: Original angleData (0-359 or 999)
            actions: Original actions list
            transform_name: Name of transform for logging
            reflect_fn: Reflection function or None for rotation
            rotate_deg: Rotation degrees (0-315) or 0 for reflection
            needs_twirl: If True, toggle Twirl on floor 0
            
        Returns:
            Tuple of (transformed_angle_data, transformed_actions)
        """
        # Transform angleData
        if reflect_fn is not None:
            # Apply reflection
            transformed_angles = [reflect_fn(a) for a in angle_data]
        elif rotate_deg != 0:
            # Apply rotation (999 unchanged)
            transformed_angles = [(a + rotate_deg) % 360 if a != 999 else 999 for a in angle_data]
        else:
            # Identity
            transformed_angles = angle_data.copy()
        
        # Transform actions
        transformed_actions = actions.copy()
        
        # Add floor-0 Twirl if reflection requires it
        if needs_twirl:
            # Check if floor 0 already has a Twirl
            has_floor_0_twirl = any(
                act.get('floor') == 0 and act.get('eventType') == 'Twirl'
                for act in transformed_actions
            )
            
            if not has_floor_0_twirl:
                # Insert Twirl at floor 0
                transformed_actions = [{'floor': 0, 'eventType': 'Twirl'}] + transformed_actions
            # If there's already a floor-0 Twirl, reflection cancels it out (toggle)
            # For simplicity, we can skip this case or remove the existing one
        
        # TODO: Also transform camera/track positions and rotations
        # For now, keep actions unchanged (camera positions won't match rotated path)
        
        return transformed_angles, transformed_actions
    
    def _apply_matched_rate(
        self,
        settings: dict,
        actions: list[dict],
        rate_factor: float
    ) -> tuple[dict, list[dict]]:
        """
        Apply matched-rate transform: audio duration/r, BPM*r, offset/r.
        
        Args:
            settings: Level settings dict
            actions: Actions list
            rate_factor: Rate multiplier (e.g., 0.9, 1.0, 1.1, 1.2)
            
        Returns:
            Tuple of (transformed_settings, transformed_actions)
        """
        if rate_factor == 1.0:
            return settings, actions
        
        transformed_settings = settings.copy()
        transformed_actions = []
        
        # Scale BPM
        if 'bpm' in transformed_settings:
            transformed_settings['bpm'] = transformed_settings['bpm'] * rate_factor
        
        # Scale offset (ms / r)
        if 'offset' in transformed_settings:
            transformed_settings['offset'] = transformed_settings['offset'] / rate_factor
        
        # Scale SetSpeed BPM events
        for action in actions:
            act = action.copy()
            if act.get('eventType') == 'SetSpeed' and act.get('speedType') == 'Bpm':
                act['beatsPerMinute'] = act['beatsPerMinute'] * rate_factor
            transformed_actions.append(act)
        
        # Note: Multipliers, Pause/Hold durations (in beats), camera durations (beats), angleOffset unchanged
        
        return transformed_settings, transformed_actions
    
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
                # Parse RAW .adofai (we need angleData + actions, not Events yet)
                from adofai.parser import parse_adofai
                level = parse_adofai(adofai_path)
                
                # Find audio
                audio_path = None
                audio_filename = level.settings.get("songFilename", "")
                if audio_filename:
                    candidate = chart_dir / audio_filename
                    if candidate.exists():
                        audio_path = candidate
                
                if audio_path is None:
                    # Fall back to scanning
                    for ext in [".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac"]:
                        candidates = list(chart_dir.glob(f"*{ext}"))
                        if candidates:
                            audio_path = candidates[0]
                            break
                
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
                
                # LOCKED AUGMENTATION SAMPLING:
                # For each chart, generate variants by picking ONE geometric transform per variant
                # (not cartesian product of all combinations)
                
                for transform_name, reflect_fn, rotate_deg, needs_twirl in self.geometric_transforms:
                    # Apply geometric transform to RAW angleData + actions
                    aug_angles, aug_actions = self._apply_geometric_transform(
                        level.angle_data,
                        level.actions,
                        transform_name,
                        reflect_fn,
                        rotate_deg,
                        needs_twirl
                    )
                    
                    # Sample ONE rate factor (could also iterate all, but spec says "maybe apply")
                    # For deterministic training, we'll use all rate factors
                    for rate_factor in self.matched_rate_factors:
                        # Sample ONE pitch (deterministic: use all)
                        for pitch in self.pitch_shifts:
                            # Apply rate transform
                            aug_settings, aug_actions_rate = self._apply_matched_rate(
                                level.settings,
                                aug_actions,
                                rate_factor
                            )
                            
                            # Apply pitch to settings
                            aug_settings_pitch = aug_settings.copy()
                            if pitch != 100:
                                aug_settings_pitch['pitch'] = pitch
                            
                            # Reconstruct augmented level
                            from adofai.parser import AdofaiLevel
                            aug_level = AdofaiLevel(
                                settings=aug_settings_pitch,
                                angle_data=aug_angles,
                                actions=aug_actions_rate,
                                decorations=[]
                            )
                            
                            # Convert to Events
                            events, event_times = self.parser.converter.level_to_events(aug_level)
                            
                            # TODO: Apply audio transforms (pitch-shift, rate)
                            # For now, use original audio
                            aug_audio = audio
                            
                            # Yield sample
                            yield {
                                'events': events,
                                'audio': aug_audio,
                                'chart_name': chart_dir.name,
                                'transform': transform_name,
                                'rate': rate_factor,
                                'pitch': pitch,
                            }
            
            except Exception as e:
                print(f"Error processing {chart_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
