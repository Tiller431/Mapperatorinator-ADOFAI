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
import torchaudio
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
        
        # Lossless augmentation: CONTINUOUS UNIFORM SAMPLING (independent transforms)
        # Transforms are INDEPENDENT (rotate AND reflect MAY both apply, not XOR)
        
        # Rotation: sample R ~ Uniform[0, 360) with probability p_rotate
        self.p_rotate = getattr(args, 'adofai_rotate_prob', 1.0)  # Default: always rotate
        
        # Reflection: with p_reflect, pick one axis from proven family
        self.p_reflect = getattr(args, 'adofai_reflect_prob', 0.5)
        self.reflect_axes = [
            ('x_flip', lambda a: (-a) % 360 if a != 999 else 999),
            ('y_flip', lambda a: (180 - a) % 360 if a != 999 else 999),
            ('diag_y_eq_x', lambda a: (90 - a) % 360 if a != 999 else 999),
            ('diag_y_eq_neg_x', lambda a: (270 - a) % 360 if a != 999 else 999),
        ]
        
        # Same-duration pitch: with p_pitch, sample settings.pitch ~ Uniform[80, 120]
        self.p_pitch = getattr(args, 'adofai_pitch_prob', 0.5)
        self.pitch_range = getattr(args, 'adofai_pitch_range', [80, 120])
        
        # Matched-rate: with p_rate, sample r ~ Uniform[0.85, 1.25]
        self.p_rate = getattr(args, 'adofai_rate_prob', 0.5)
        self.rate_range = getattr(args, 'adofai_rate_range', [0.85, 1.25])
        
        print(f"ADOFAI lossless augmentation (continuous uniform, independent):")
        print(f"  Rotate: p={self.p_rotate}, R ~ Uniform[0, 360)")
        print(f"  Reflect: p={self.p_reflect}, axis ~ {{X, Y, y=x, y=-x}}")
        print(f"  Pitch: p={self.p_pitch}, settings.pitch ~ Uniform{self.pitch_range}")
        print(f"  Rate: p={self.p_rate}, r ~ Uniform{self.rate_range}")
        print(f"  Transforms are INDEPENDENT (not XOR or cartesian)")
        
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
        print(f"Continuous uniform augmentation enabled (independent transforms)")
        
        return chart_dirs
    
    def _apply_rotation(
        self,
        angle_data: list[int],
        actions: list[dict],
        rotate_deg: float
    ) -> tuple[list[int], list[dict]]:
        """
        Apply rotation to RAW angleData AND camera/track world positions + rotations.
        
        Args:
            angle_data: Original angleData (0-359 or 999)
            actions: Original actions
            rotate_deg: Rotation degrees (continuous, sampled from Uniform[0, 360))
            
        Returns:
            Tuple of (rotated_angleData, rotated_actions)
        """
        # Rotate tile angles
        rotated_angles = [(int(a + rotate_deg) % 360) if a != 999 else 999 for a in angle_data]
        
        # Rotate camera/track world positions and rotations
        # angleOffset is NOT rotated (it's a floor-relative offset)
        rotated_actions = []
        for action in actions:
            act = action.copy()
            event_type = act.get('eventType', '')
            
            # For camera and track moves, rotate world positions and rotations
            if event_type in ('MoveCamera', 'PositionTrack', 'AnimateTrack'):
                # Rotate position vector (x, y) by rotate_deg
                pos = act.get('position', [0, 0])
                if len(pos) == 2:
                    x, y = pos[0], pos[1]
                    rad = rotate_deg * (3.14159265359 / 180.0)
                    cos_r = np.cos(rad)
                    sin_r = np.sin(rad)
                    new_x = x * cos_r - y * sin_r
                    new_y = x * sin_r + y * cos_r
                    act['position'] = [new_x, new_y]
                
                # Rotate rotation field
                if 'rotation' in act:
                    act['rotation'] = (act['rotation'] + rotate_deg) % 360
            
            elif event_type == 'MoveTrack':
                # MoveTrack uses positionOffset, not position
                pos_offset = act.get('positionOffset', [0, 0])
                if len(pos_offset) == 2:
                    x, y = pos_offset[0], pos_offset[1]
                    rad = rotate_deg * (3.14159265359 / 180.0)
                    cos_r = np.cos(rad)
                    sin_r = np.sin(rad)
                    new_x = x * cos_r - y * sin_r
                    new_y = x * sin_r + y * cos_r
                    act['positionOffset'] = [new_x, new_y]
            
            rotated_actions.append(act)
        
        return rotated_angles, rotated_actions
    
    def _apply_reflection(
        self,
        angle_data: list[int],
        actions: list[dict],
        reflect_fn
    ) -> tuple[list[int], list[dict]]:
        """
        Apply reflection to RAW angleData and add floor-0 Twirl.
        
        Args:
            angle_data: Original angleData
            actions: Original actions
            reflect_fn: Reflection function from proven family
            
        Returns:
            Tuple of (reflected_angles, actions_with_twirl)
        """
        reflected_angles = [reflect_fn(a) for a in angle_data]
        
        # Toggle floor-0 Twirl
        has_floor_0_twirl = any(
            act.get('floor') == 0 and act.get('eventType') == 'Twirl'
            for act in actions
        )
        
        if not has_floor_0_twirl:
            # Add floor-0 Twirl
            actions_with_twirl = [{'floor': 0, 'eventType': 'Twirl'}] + list(actions)
        else:
            # Remove floor-0 Twirl (toggle)
            actions_with_twirl = [
                act for act in actions
                if not (act.get('floor') == 0 and act.get('eventType') == 'Twirl')
            ]
        
        return reflected_angles, actions_with_twirl
    
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
                
                # CONTINUOUS UNIFORM SAMPLING (independent transforms)
                # Each chart yields ONE augmented variant per epoch
                # Transforms are sampled independently (not XOR)
                
                aug_angles = level.angle_data.copy()
                aug_actions = level.actions.copy()
                aug_settings = level.settings.copy()
                
                transform_desc = []
                
                # 1. Rotation: with p_rotate, sample R ~ Uniform[0, 360)
                if random.random() < self.p_rotate:
                    rotate_deg = random.uniform(0, 360)
                    aug_angles, aug_actions = self._apply_rotation(aug_angles, aug_actions, rotate_deg)
                    transform_desc.append(f'R={rotate_deg:.1f}°')
                
                # 2. Reflection: with p_reflect, pick one axis
                if random.random() < self.p_reflect:
                    axis_name, reflect_fn = random.choice(self.reflect_axes)
                    aug_angles, aug_actions = self._apply_reflection(aug_angles, aug_actions, reflect_fn)
                    transform_desc.append(f'reflect_{axis_name}')
                
                # 3. Matched-rate: with p_rate, sample r ~ Uniform[rate_range]
                rate_factor = 1.0
                if random.random() < self.p_rate:
                    rate_factor = random.uniform(self.rate_range[0], self.rate_range[1])
                    aug_settings, aug_actions = self._apply_matched_rate(aug_settings, aug_actions, rate_factor)
                    transform_desc.append(f'rate={rate_factor:.3f}')
                
                # 4. Same-duration pitch: with p_pitch, sample settings.pitch ~ Uniform[pitch_range]
                if random.random() < self.p_pitch:
                    pitch = random.uniform(self.pitch_range[0], self.pitch_range[1])
                    aug_settings['pitch'] = int(pitch)
                    transform_desc.append(f'pitch={int(pitch)}')
                else:
                    aug_settings['pitch'] = 100
                
                # Reconstruct augmented level
                from adofai.parser import AdofaiLevel
                aug_level = AdofaiLevel(
                    settings=aug_settings,
                    angle_data=aug_angles,
                    actions=aug_actions,
                    decorations=[]
                )
                
                # Convert to Events
                events, event_times = self.parser.converter.level_to_events(aug_level)
                
                # Apply audio transforms
                aug_audio = audio
                sample_rate = self.args.sample_rate if hasattr(self.args, 'sample_rate') else 16000
                
                # Apply pitch shift (same duration)
                if aug_settings.get('pitch', 100) != 100:
                    pitch_shift_factor = aug_settings['pitch'] / 100.0  # 80-120 → 0.8-1.2
                    # Pitch shift using resampling: shift pitch without changing duration
                    # This is "same-duration pitch" as required
                    n_steps = 12 * np.log2(pitch_shift_factor)  # Convert to semitones
                    effects = [
                        ["pitch", str(int(n_steps * 100))],  # Pitch shift in cents
                        ["rate", str(sample_rate)],  # Keep original sample rate
                    ]
                    try:
                        aug_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                            aug_audio.unsqueeze(0) if aug_audio.dim() == 1 else aug_audio,
                            sample_rate,
                            effects
                        )
                        aug_audio = aug_audio.squeeze(0)
                    except Exception as e:
                        print(f"Warning: Pitch shift failed: {e}, using original audio")
                
                # Apply matched-rate transform (changes duration)
                if rate_factor != 1.0:
                    # Time-stretch audio by factor 1/rate_factor
                    # rate_factor=1.2 → audio becomes 1/1.2=0.833x duration (faster)
                    # BPM is already scaled by rate_factor, so audio must be (duration / rate_factor)
                    try:
                        stretch_factor = 1.0 / rate_factor
                        effects = [
                            ["tempo", str(stretch_factor)],  # Stretch time without pitch change
                        ]
                        aug_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                            aug_audio.unsqueeze(0) if aug_audio.dim() == 1 else aug_audio,
                            sample_rate,
                            effects
                        )
                        aug_audio = aug_audio.squeeze(0)
                    except Exception as e:
                        print(f"Warning: Rate transform failed: {e}, using original audio")
                
                # Yield sample
                yield {
                    'events': events,
                    'audio': aug_audio,
                    'chart_name': chart_dir.name,
                    'transforms': ', '.join(transform_desc) if transform_desc else 'identity',
                }
            
            except Exception as e:
                print(f"Error processing {chart_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
