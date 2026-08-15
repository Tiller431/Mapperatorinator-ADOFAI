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


def load_audio_file(
    file: str | Path,
    sample_rate: int = 16000,
    normalize: bool = True,
    max_duration_sec: Optional[float] = 60.0,
) -> npt.NDArray:
    """
    Load an audio file as a numpy time-series array.
    
    Uses pydub to handle multiple formats and resamples to target sample rate.
    
    Args:
        file: Path to audio file
        sample_rate: Target sample rate
        normalize: Whether to normalize amplitude
        max_duration_sec: Maximum duration in seconds (crops from center if longer)
    
    Returns:
        Audio samples as float32 numpy array
    """
    from pydub import AudioSegment
    
    file = Path(file)
    audio = AudioSegment.from_file(file)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)  # Mono
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Crop to max duration (center crop for long songs)
    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sample_rate)
        if len(samples) > max_samples:
            # Center crop
            start = (len(samples) - max_samples) // 2
            samples = samples[start:start + max_samples]
    
    if normalize:
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples *= 1.0 / max_val
    return samples


def compute_log_mel_spectrogram(
    audio: npt.NDArray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
) -> npt.NDArray:
    """
    Compute log-mel spectrogram from audio waveform.
    
    This is a compact audio representation suitable for neural network input,
    reducing memory usage from O(audio_samples) to O(time_frames * n_mels).
    
    Args:
        audio: Audio waveform [samples]
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel filterbanks
    
    Returns:
        Log-mel spectrogram [time_frames, n_mels]
    """
    try:
        import torchaudio.transforms as T
        import torch
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        # Ensure 1D
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.squeeze(0)
        
        # Compute mel spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_spec = mel_transform(audio_tensor)  # [n_mels, time_frames]
        
        # Convert to log scale (add small epsilon for numerical stability)
        log_mel_spec = torch.log(mel_spec + 1e-9)
        
        # Transpose to [time_frames, n_mels] for easier processing
        log_mel_spec = log_mel_spec.transpose(0, 1)
        
        return log_mel_spec.numpy()
        
    except ImportError:
        # Fallback: simple mean-pooling (less optimal but works without torchaudio)
        # Chunk audio into fixed-size frames and compute mean
        chunk_size = hop_length
        n_frames = len(audio) // chunk_size
        
        # Reshape and average
        audio_chunked = audio[:n_frames * chunk_size].reshape(n_frames, chunk_size)
        features = audio_chunked.mean(axis=1, keepdims=True)  # [n_frames, 1]
        
        # Expand to n_mels channels by repeating (crude but functional)
        features = np.tile(features, (1, n_mels))  # [n_frames, n_mels]
        
        return features


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
        """
        Find audio file in chart directory.
        
        Tries multiple strategies:
        1. Check songFilename from level.adofai settings
        2. Scan for common audio extensions (case-insensitive)
        """
        # Strategy 1: Try songFilename from level.adofai
        try:
            from .parser import parse_adofai
            level = parse_adofai(self.level_path)
            song_filename = level.settings.get('songFilename', '')
            if song_filename:
                # Try as relative path from chart_dir
                audio_path = chart_dir / song_filename
                if audio_path.exists() and audio_path.is_file():
                    return audio_path
                
                # Try just the basename (in case path is different)
                from pathlib import PurePath
                basename = PurePath(song_filename).name
                audio_path = chart_dir / basename
                if audio_path.exists() and audio_path.is_file():
                    return audio_path
        except Exception:
            # If level parsing fails, fall through to scan
            pass
        
        # Strategy 2: Scan directory for audio files (case-insensitive)
        audio_extensions = {'.ogg', '.mp3', '.wav', '.flac', '.m4a', '.aac'}
        
        for file in chart_dir.iterdir():
            if file.is_file():
                # Check extension case-insensitively
                if file.suffix.lower() in audio_extensions:
                    return file
        
        return None
    
    def has_audio(self) -> bool:
        """Check if audio file exists."""
        return self.audio_file is not None and self.audio_file.exists()
    
    def load_level(self) -> AdofaiLevel:
        """Parse the ADOFAI level."""
        return parse_adofai(self.level_path)
    
    def load_audio(
        self,
        sample_rate: int = 16000,
        max_duration_sec: Optional[float] = 60.0,
    ) -> Optional[npt.NDArray]:
        """Load audio file."""
        if not self.has_audio():
            return None
        return load_audio_file(
            self.audio_file,
            sample_rate=sample_rate,
            max_duration_sec=max_duration_sec,
        )


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
        max_audio_duration_sec: float = 60.0,
        use_spectrogram: bool = True,
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
            max_audio_duration_sec: Max audio duration in seconds (center-cropped if longer)
            use_spectrogram: Whether to convert audio to log-mel spectrogram (recommended)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_split = train_split
        self.seed = seed
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.skip_invalid = skip_invalid
        self.max_audio_duration_sec = max_audio_duration_sec
        self.use_spectrogram = use_spectrogram
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
                
                # Load audio (with duration cap)
                audio = entry.load_audio(
                    self.sample_rate,
                    max_duration_sec=self.max_audio_duration_sec,
                )
                if audio is None and not self.skip_invalid:
                    raise ValueError(f"No audio for {entry.chart_name}")
                
                if audio is None:
                    continue
                
                # Convert to spectrogram if requested
                if self.use_spectrogram:
                    audio = compute_log_mel_spectrogram(audio, self.sample_rate)
                
                # Create sample
                sample = {
                    'chart_name': entry.chart_name,
                    'workshop_id': entry.workshop_id,
                    'audio': audio,
                    'events': events,
                    'event_times': event_times,
                    'bpm': level.settings.get('bpm', 120),
                    'offset': level.settings.get('offset', 0),
                    'is_spectrogram': self.use_spectrogram,
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
    is_spectrogram = batch[0].get('is_spectrogram', False)
    
    # Pad audio to max length in batch
    if is_spectrogram:
        # Audio is [time_frames, n_mels]
        max_audio_len = max(item['audio'].shape[0] for item in batch)
        n_mels = batch[0]['audio'].shape[1]
        audio_batch = []
        for item in batch:
            audio = item['audio']
            pad_width = ((0, max_audio_len - audio.shape[0]), (0, 0))
            padded = np.pad(audio, pad_width, mode='constant')
            audio_batch.append(padded)
        audio_batch = torch.tensor(np.stack(audio_batch), dtype=torch.float32)
    else:
        # Audio is [samples] (raw waveform)
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
        'is_spectrogram': is_spectrogram,
    }
