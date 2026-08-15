"""
ADOFAI dataset for osuT5 Whisper training.

Yields raw waveform frames `[src_seq_len-1, hop_length]` (flattened) like
MmrsDataset. The model owns the Mel spectrogram. Augmentations run on raw
angleData + actions, then convert to osuT5 Events.
"""

from __future__ import annotations

import random
from multiprocessing.managers import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import IterableDataset

from .data_utils import SequenceDatasetMixin, get_song_length, load_audio_file
from .adofai_parser import AdofaiParser
from ..tokenizer import Tokenizer, ContextType
from ..config import DataConfig
from adofai.parser import AdofaiLevel, parse_adofai
from .adofai_augment import (
    ADOFAI_DIFFICULTY_PROXY,
    REFLECT_AXES,
    apply_matched_rate,
    apply_reflection,
    apply_rotation,
    resolve_difficulty,
)


def _pitch_shift_same_duration(samples: npt.NDArray, sample_rate: int, pitch: float) -> npt.NDArray:
    """Same-duration waveform pitch shift. Chart events stay untouched."""
    if abs(pitch - 100.0) < 1e-6:
        return samples
    n_steps = float(12.0 * np.log2(pitch / 100.0))
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    try:
        shifted = torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            scale_factor=100.0 / pitch,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        # interpolate changes duration; resample back to original length
        shifted = torch.nn.functional.interpolate(
            shifted.unsqueeze(0),
            size=waveform.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        return shifted.squeeze(0).numpy()
    except Exception as exc:
        print(f"Warning: same-duration pitch shift failed ({exc}); using original audio")
        return samples


class AdofaiDataset(IterableDataset, SequenceDatasetMixin):
    """ADOFAI IterableDataset with lossless augs + Mmrs-style frame windowing."""

    def __init__(
        self,
        args: DataConfig,
        parser: AdofaiParser,
        tokenizer: Tokenizer,
        chart_dirs: Optional[list[Path]] = None,
        test: bool = False,
        shared: Namespace = None,
        **kwargs,
    ):
        super().__init__()
        self.path = Path(args.test_dataset_path if test else args.train_dataset_path)
        self.start = args.test_dataset_start if test else args.train_dataset_start
        self.end = args.test_dataset_end if test else args.train_dataset_end
        self.args = args
        self.parser = parser
        self.tokenizer = tokenizer
        self.test = test
        self.shared = shared
        self.frame_seq_len = args.src_seq_len - 1
        self.min_pre_token_len = 4
        self.pre_token_len = args.tgt_seq_len // 2
        self.add_pre_tokens = args.add_pre_tokens
        self.add_empty_sequences = args.add_empty_sequences

        self.p_rotate = getattr(args, "adofai_rotate_prob", 1.0)
        self.p_reflect = getattr(args, "adofai_reflect_prob", 0.5)
        self.p_pitch = getattr(args, "adofai_pitch_prob", 0.5)
        self.pitch_range = getattr(args, "adofai_pitch_range", [80, 120])
        self.p_rate = getattr(args, "adofai_rate_prob", 0.5)
        self.rate_range = getattr(args, "adofai_rate_range", [0.85, 1.25])
        self.default_difficulty = getattr(args, "adofai_default_difficulty", ADOFAI_DIFFICULTY_PROXY)
        self.reflect_axes = list(REFLECT_AXES.keys())

        if chart_dirs is not None:
            self.chart_dirs = chart_dirs
        else:
            self.chart_dirs = self._find_chart_dirs()

        print("ADOFAI lossless augmentation (continuous uniform, independent):")
        print(f"  Rotate: p={self.p_rotate}, R ~ Uniform[0, 360)")
        print(f"  Reflect: p={self.p_reflect}, axis ~ {{X, Y, y=x, y=-x}}")
        print(f"  Pitch: p={self.p_pitch}, settings.pitch ~ Uniform{self.pitch_range}")
        print(f"  Rate: p={self.p_rate}, r ~ Uniform{self.rate_range} (audio duration/r)")
        print(f"  Frames: [{self.frame_seq_len}, hop={args.hop_length}] (model owns Mel)")

    def _find_chart_dirs(self) -> list[Path]:
        chart_dirs = []
        if not self.path.exists():
            print(f"Warning: Dataset path does not exist: {self.path}")
            return chart_dirs
        for item in self.path.iterdir():
            if item.is_dir() and (item / "level.adofai").exists():
                chart_dirs.append(item)
        chart_dirs = sorted(chart_dirs)
        if self.end > 0:
            chart_dirs = chart_dirs[self.start:self.end]
        else:
            chart_dirs = chart_dirs[self.start:]
        print(f"Found {len(chart_dirs)} ADOFAI charts in {self.path}")
        return chart_dirs

    def _apply_rotation(self, angle_data, actions, rotate_deg):
        return apply_rotation(angle_data, actions, rotate_deg)

    def _apply_reflection(self, angle_data, actions, axis):
        return apply_reflection(angle_data, actions, axis)

    def _apply_matched_rate(self, settings, actions, rate_factor):
        return apply_matched_rate(settings, actions, rate_factor)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            charts = [c for i, c in enumerate(self.chart_dirs) if i % worker_info.num_workers == worker_info.id]
        else:
            charts = list(self.chart_dirs)
        if not self.test:
            random.shuffle(charts)
        for chart_dir in charts:
            yield from self._iter_chart(chart_dir)

    def _iter_chart(self, chart_dir: Path):
        adofai_path = chart_dir / "level.adofai"
        try:
            level = parse_adofai(adofai_path)
        except Exception as exc:
            print(f"Error parsing {chart_dir.name}: {exc}")
            return

        audio_path = self.parser.find_audio_path(adofai_path, level)
        if audio_path is None:
            print(f"Warning: No audio found for {chart_dir.name}")
            return

        aug_angles = list(level.angle_data)
        aug_actions = [dict(a) for a in level.actions]
        aug_settings = dict(level.settings)

        if not self.test and random.random() < self.p_rotate:
            rotate_deg = random.uniform(0, 360)
            aug_angles, aug_actions = apply_rotation(aug_angles, aug_actions, rotate_deg)
        if not self.test and random.random() < self.p_reflect:
            axis = random.choice(self.reflect_axes)
            aug_angles, aug_actions = apply_reflection(aug_angles, aug_actions, axis)

        rate_factor = 1.0
        if not self.test and random.random() < self.p_rate:
            rate_factor = random.uniform(self.rate_range[0], self.rate_range[1])
            aug_settings, aug_actions = apply_matched_rate(aug_settings, aug_actions, rate_factor)

        if not self.test and random.random() < self.p_pitch:
            pitch = random.uniform(self.pitch_range[0], self.pitch_range[1])
            aug_settings["pitch"] = int(pitch)
        else:
            pitch = float(aug_settings.get("pitch", 100) or 100)
            aug_settings["pitch"] = int(pitch)

        try:
            # Matched-rate: load_audio_file speed=r makes duration become duration/r
            audio_samples = load_audio_file(
                str(audio_path),
                self.args.sample_rate,
                speed=rate_factor,
                normalize=getattr(self.args, "normalize_audio", True),
            )
        except Exception as exc:
            print(f"Warning: Failed to load audio {audio_path}: {exc}")
            return

        if float(aug_settings.get("pitch", 100)) != 100:
            audio_samples = _pitch_shift_same_duration(
                audio_samples, self.args.sample_rate, float(aug_settings["pitch"])
            )

        aug_level = AdofaiLevel(
            settings=aug_settings,
            angle_data=aug_angles,
            actions=aug_actions,
            decorations=[],
        )
        difficulty = resolve_difficulty(chart_dir, aug_settings, self.default_difficulty)
        song_length = get_song_length(audio_samples, self.args.sample_rate)
        frames, frame_times = self._get_frames(audio_samples)

        context_info = {"in": [ContextType.NONE], "out": [ContextType.TIMING, ContextType.MAP]}
        if self.args.context_types:
            context_info = random.choices(self.args.context_types, weights=self.args.context_weights)[0]
            context_info = context_info.copy()

        def get_context(context: ContextType, identifier: str, add_type: bool = True) -> dict:
            data = {"extra": {"context_type": context, "add_type": add_type, "id": f"{identifier}_{context.value}"}}
            if context == ContextType.NONE:
                data["events"], data["event_times"] = [], []
            elif context == ContextType.TIMING:
                data["events"], data["event_times"] = self.parser.parse_timing(aug_level)
            else:
                data["events"], data["event_times"] = self.parser.parse(aug_level)
            return data

        extra_data = {
            "beatmap_idx": torch.tensor(0, dtype=torch.long),
            "mapper_idx": torch.tensor(0, dtype=torch.long),
            "difficulty": torch.tensor(difficulty, dtype=torch.float32),
            "special": {
                "beatmap_id": abs(hash(chart_dir.name)) % 10_000_000,
                "difficulty": difficulty,
                "song_length": song_length,
            },
        }
        out_context = [
            get_context(context, "out", add_type=self.args.add_out_context_types)
            for context in context_info["out"]
        ]
        in_context = [get_context(context, "in") for context in context_info["in"]]
        sequences = self._create_sequences(frames, frame_times, out_context, in_context, extra_data)
        yield from self.process_sequences(sequences, adofai_path)
