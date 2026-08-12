"""
ADOFAI Dataset for training.

Loads ADOFAI charts from Workshop-style directories and prepares them for training.
Dataset layout:
    <data_root>/<workshopId>__<chartName>/
        level.adofai
        <audio>.ogg|.mp3|.wav|.flac
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional
import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import IterableDataset

from .parser import parse_adofai, AdofaiLevel
from .converter import AdofaiConverter
from .event import AdofaiEvent


def load_audio_file(file: str | Path, sample_rate: int = 16000, normalize: bool = True) -> npt.NDArray:
    """
    Load an audio file as a numpy time-series array.
    
    Uses pydub to handle multiple formats and resamples to target sample rate.
    """
    from pydub import AudioSegment
    
    file = Path(file)
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)  # Mono
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if normalize:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples *= 1.0 / max_val
    return samples


class AdofaiDatasetEntry:
    """Single entry in the ADOFAI dataset."""
    
    def __init__(
        self,
        chart_dir: Path,
        workshop_id: Optional[str] = None,
        audio_file: Optional[Path] = None,
    ):
        self.chart_dir = chart_dir
        self.workshop_id = workshop_id or chart_dir.name.split("__")[0]
        self.chart_name = chart_dir.name
        
        # Find level.adofai
        self.level_path = chart_dir / "level.adofai"
        if not self.level_path.exists():
            raise FileNotFoundError(f"level.adofai not found in {chart_dir}")
        
        # Find audio file (try provided path or search in directory)
        if audio_file and audio_file.exists():
            self.audio_file = audio_file
        else:
            self.audio_file = self._find_audio_file(chart_dir)
    
    def _find_audio_file(self, chart_dir: Path) -> Optional[Path]:
        """Find audio file in chart directory."""
        audio_extensions = ['.ogg', '.mp3', '.wav', '.flac', '.m4a']
        for ext in audio_extensions:
            audio_files = list(chart_dir.glob(f"*{ext}"))
            if audio_files:
                return audio_files[0]
        return None
    
    def has_audio(self) -> bool:
        """Check if audio file exists."""
        return self.audio_file is not None and self.audio_file.exists()
    
    def load_level(self) -> AdofaiLevel:
        """Parse the ADOFAI level."""
        return parse_adofai(self.level_path)
    
    def load_audio(self, sample_rate: int = 16000) -> Optional[npt.NDArray]:
        """Load audio file."""
        if not self.has_audio():
            return None
        return load_audio_file(self.audio_file, sample_rate=sample_rate)


class AdofaiDataset(IterableDataset):
    """
    PyTorch IterableDataset for ADOFAI charts.
    
    Loads charts from Workshop-style directories and converts them to
    training samples (audio + event sequences).
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        index_json: Optional[str | Path] = None,
        split: str = "train",
        train_split: float = 0.9,
        seed: int = 42,
        sample_rate: int = 16000,
        max_samples: Optional[int] = None,
        skip_invalid: bool = True,
    ):
        """
        Initialize ADOFAI dataset.
        
        Args:
            data_dir: Root directory containing chart folders
            index_json: Optional JSON index with chart metadata
            split: "train" or "val"
            train_split: Fraction of data for training (default 0.9)
            seed: Random seed for splitting
            sample_rate: Audio sample rate
            max_samples: Maximum number of samples to load (for smoke testing)
            skip_invalid: Skip charts with missing/broken audio
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_split = train_split
        self.seed = seed
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.skip_invalid = skip_invalid
        self.converter = AdofaiConverter()
        
        # Load entries
        if index_json and Path(index_json).exists():
            self.entries = self._load_from_index(index_json)
        else:
            self.entries = self._scan_directory()
        
        # Split train/val
        self.entries = self._split_entries(self.entries, split, train_split, seed)
        
        if max_samples:
            self.entries = self.entries[:max_samples]
        
        print(f"Loaded {len(self.entries)} {split} entries from {self.data_dir}")
    
    def _scan_directory(self) -> list[AdofaiDatasetEntry]:
        """Scan data_dir for chart folders."""
        entries = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        for chart_dir in sorted(self.data_dir.iterdir()):
            if not chart_dir.is_dir():
                continue
            
            # Check for level.adofai
            level_path = chart_dir / "level.adofai"
            if not level_path.exists():
                continue
            
            try:
                entry = AdofaiDatasetEntry(chart_dir)
                if self.skip_invalid and not entry.has_audio():
                    print(f"Skipping {chart_dir.name}: no audio found")
                    continue
                entries.append(entry)
            except Exception as e:
                print(f"Error loading {chart_dir.name}: {e}")
                if not self.skip_invalid:
                    raise
        
        return entries
    
    def _load_from_index(self, index_json: str | Path) -> list[AdofaiDatasetEntry]:
        """Load entries from JSON index."""
        with open(index_json, 'r') as f:
            index_data = json.load(f)
        
        entries = []
        for item in index_data:
            chart_dir = self.data_dir / item['chart_dir']
            audio_file = Path(item['audio']) if 'audio' in item else None
            
            try:
                entry = AdofaiDatasetEntry(
                    chart_dir=chart_dir,
                    workshop_id=item.get('workshop_id'),
                    audio_file=audio_file,
                )
                
                if self.skip_invalid and not entry.has_audio():
                    print(f"Skipping {chart_dir.name}: no audio")
                    continue
                
                entries.append(entry)
            except Exception as e:
                print(f"Error loading {chart_dir.name}: {e}")
                if not self.skip_invalid:
                    raise
        
        return entries
    
    def _split_entries(
        self,
        entries: list[AdofaiDatasetEntry],
        split: str,
        train_split: float,
        seed: int
    ) -> list[AdofaiDatasetEntry]:
        """Split entries into train/val."""
        rng = random.Random(seed)
        entries_copy = entries.copy()
        rng.shuffle(entries_copy)
        
        n_train = int(len(entries_copy) * train_split)
        
        if split == "train":
            return entries_copy[:n_train]
        else:  # val
            return entries_copy[n_train:]
    
    def __iter__(self):
        """Iterate over dataset entries."""
        entries = self.entries.copy()
        
        # Shuffle for training
        if self.split == "train":
            random.shuffle(entries)
        
        for entry in entries:
            try:
                # Load level and convert to events
                level = entry.load_level()
                events, event_times = self.converter.level_to_events(level)
                
                # Load audio
                audio = entry.load_audio(self.sample_rate)
                if audio is None and not self.skip_invalid:
                    raise ValueError(f"No audio for {entry.chart_name}")
                
                if audio is None:
                    continue
                
                # Create sample
                sample = {
                    'chart_name': entry.chart_name,
                    'workshop_id': entry.workshop_id,
                    'audio': audio,
                    'events': events,
                    'event_times': event_times,
                    'bpm': level.settings.get('bpm', 120),
                    'offset': level.settings.get('offset', 0),
                }
                
                yield sample
                
            except Exception as e:
                print(f"Error processing {entry.chart_name}: {e}")
                if not self.skip_invalid:
                    raise


def collate_adofai_batch(batch):
    """
    Collate function for ADOFAI dataset batches.
    
    Handles variable-length sequences by padding.
    """
    if len(batch) == 0:
        return None
    
    # Extract fields
    chart_names = [item['chart_name'] for item in batch]
    workshop_ids = [item['workshop_id'] for item in batch]
    bpms = torch.tensor([item['bpm'] for item in batch], dtype=torch.float32)
    offsets = torch.tensor([item['offset'] for item in batch], dtype=torch.float32)
    
    # Pad audio to max length in batch
    max_audio_len = max(len(item['audio']) for item in batch)
    audio_batch = []
    for item in batch:
        audio = item['audio']
        padded = np.pad(audio, (0, max_audio_len - len(audio)), mode='constant')
        audio_batch.append(padded)
    audio_batch = torch.tensor(np.stack(audio_batch), dtype=torch.float32)
    
    # Collect events (keep as list for now, tokenization happens later)
    events_batch = [item['events'] for item in batch]
    event_times_batch = [item['event_times'] for item in batch]
    
    return {
        'chart_names': chart_names,
        'workshop_ids': workshop_ids,
        'audio': audio_batch,
        'events': events_batch,
        'event_times': event_times_batch,
        'bpm': bpms,
        'offset': offsets,
    }
