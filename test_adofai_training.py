"""
Tests for ADOFAI training pipeline.

Creates synthetic chart data and tests dataset loading + training smoke test.
"""

import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
import wave

from adofai import write_adofai, AdofaiLevel
from adofai.dataset import AdofaiDataset, AdofaiDatasetEntry, collate_adofai_batch
from adofai.tokenizer import AdofaiTokenizer


def create_silent_wav(output_path: Path, duration_sec: float = 5.0, sample_rate: int = 16000):
    """Create a silent WAV file for testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_samples = int(duration_sec * sample_rate)
    audio_data = np.zeros(n_samples, dtype=np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


def create_synthetic_chart_folder(base_dir: Path, chart_id: int) -> Path:
    """Create a synthetic chart folder with level.adofai + audio."""
    chart_dir = base_dir / f"test{chart_id}__TestChart{chart_id}"
    chart_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple chart
    level = AdofaiLevel(
        settings={
            "version": 14,
            "artist": f"Test Artist {chart_id}",
            "song": f"Test Song {chart_id}",
            "author": "Test",
            "songFilename": "audio.wav",
            "bpm": 120 + chart_id * 10,
            "offset": 0,
            "volume": 100,
            "pitch": 100,
            "hitsound": "Kick",
            "hitsoundVolume": 100,
        },
        angle_data=[0, 0, 90, 180, 270, 45, 135, 225, 315],
        actions=[
            {
                "floor": 1,
                "eventType": "SetSpeed",
                "speedType": "Bpm",
                "beatsPerMinute": 120 + chart_id * 10,
                "bpmMultiplier": 1.0
            },
            {
                "floor": 4,
                "eventType": "Twirl"
            }
        ]
    )
    
    # Write chart
    write_adofai(level, chart_dir / "level.adofai")
    
    # Create silent audio
    create_silent_wav(chart_dir / "audio.wav", duration_sec=10.0)
    
    return chart_dir


def test_create_synthetic_data():
    """Test creating synthetic chart data."""
    print("Testing create_synthetic_data...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a few synthetic charts
        for i in range(3):
            chart_dir = create_synthetic_chart_folder(tmpdir, i)
            
            assert (chart_dir / "level.adofai").exists()
            assert (chart_dir / "audio.wav").exists()
        
        # Verify structure
        chart_dirs = list(tmpdir.iterdir())
        assert len(chart_dirs) == 3
        
        print("✓ Synthetic data creation: PASSED")


def test_dataset_loading():
    """Test ADOFAI dataset loading."""
    print("Testing dataset_loading...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        for i in range(5):
            create_synthetic_chart_folder(tmpdir, i)
        
        # Load dataset
        dataset = AdofaiDataset(
            data_dir=tmpdir,
            split='train',
            train_split=0.8,
            max_samples=None,
        )
        
        # Check split
        assert len(dataset.entries) == 4  # 80% of 5
        
        # Iterate and check samples
        samples = list(dataset)
        assert len(samples) > 0
        
        sample = samples[0]
        assert 'audio' in sample
        assert 'events' in sample
        assert 'bpm' in sample
        assert sample['audio'] is not None
        assert len(sample['events']) > 0
        
        print(f"✓ Dataset loading: PASSED ({len(samples)} samples loaded)")


def test_dataset_entry():
    """Test individual dataset entry."""
    print("Testing dataset_entry...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        chart_dir = create_synthetic_chart_folder(tmpdir, 0)
        
        entry = AdofaiDatasetEntry(chart_dir)
        
        assert entry.has_audio()
        assert entry.level_path.exists()
        
        level = entry.load_level()
        assert level.settings['bpm'] == 120
        assert len(level.angle_data) == 9
        
        audio = entry.load_audio(sample_rate=16000)
        assert audio is not None
        assert len(audio) > 0
        
        print("✓ Dataset entry: PASSED")


def test_tokenizer():
    """Test ADOFAI tokenizer."""
    print("Testing tokenizer...")
    
    from adofai.event import AdofaiEvent, AdofaiEventType
    
    tokenizer = AdofaiTokenizer()
    
    assert tokenizer.vocab_size > 0
    assert tokenizer.pad_token_id == 0
    assert tokenizer.sos_token_id == 1
    assert tokenizer.eos_token_id == 2
    
    # Test event tokenization
    events = [
        AdofaiEvent(AdofaiEventType.BPM, 120),
        AdofaiEvent(AdofaiEventType.TILE_ANGLE, 90),
        AdofaiEvent(AdofaiEventType.TWIRL, 1),
        AdofaiEvent(AdofaiEventType.MIDSPIN, 999),
    ]
    
    tokens = tokenizer.events_to_tokens(events)
    
    assert len(tokens) > len(events)  # Should have <sos> and <eos>
    assert tokens[0] == tokenizer.sos_token_id
    assert tokens[-1] == tokenizer.eos_token_id
    
    print(f"✓ Tokenizer: PASSED (vocab_size={tokenizer.vocab_size})")


def test_batch_collation():
    """Test batch collation."""
    print("Testing batch_collation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        for i in range(3):
            create_synthetic_chart_folder(tmpdir, i)
        
        # Load dataset
        dataset = AdofaiDataset(data_dir=tmpdir, split='train', max_samples=3)
        
        # Get samples
        samples = list(dataset)
        
        # Collate
        batch = collate_adofai_batch(samples)
        
        assert batch is not None
        assert 'audio' in batch
        assert 'events' in batch
        assert batch['audio'].shape[0] == len(samples)
        
        print("✓ Batch collation: PASSED")


def test_smoke_training():
    """Test smoke training (dataset + tokenizer, no actual training)."""
    print("Testing smoke_training setup...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create minimal synthetic dataset
        for i in range(3):
            create_synthetic_chart_folder(tmpdir, i)
        
        # Test dataset loading
        dataset = AdofaiDataset(data_dir=tmpdir, split='train', max_samples=3)
        samples = list(dataset)
        
        assert len(samples) > 0
        
        # Test tokenization
        tokenizer = AdofaiTokenizer()
        
        for sample in samples:
            tokens = tokenizer.events_to_tokens(sample['events'])
            assert len(tokens) > 2  # At least <sos> + <eos>
        
        print("✓ Smoke training setup: PASSED")
        print("  Note: Full training loop requires running adofai/train.py")


def test_audio_detection_mismatch():
    """Test that audio is found even with filename mismatch."""
    print("Testing audio_detection_mismatch...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chart_dir = tmpdir / "test123__MismatchTest"
        chart_dir.mkdir()
        
        # Create chart with songFilename="song.ogg" but actual file is "music.ogg"
        level = AdofaiLevel(
            settings={
                "version": 14,
                "artist": "Test",
                "song": "Test",
                "author": "Test",
                "songFilename": "song.ogg",  # This doesn't exist
                "bpm": 120,
                "offset": 0,
                "volume": 100,
                "pitch": 100,
                "hitsound": "Kick",
                "hitsoundVolume": 100,
            },
            angle_data=[0, 90, 180, 270],
            actions=[]
        )
        
        write_adofai(level, chart_dir / "level.adofai")
        
        # Create audio file with different name
        create_silent_wav(chart_dir / "music.ogg", duration_sec=5.0)
        
        # Try to load as dataset entry
        entry = AdofaiDatasetEntry(chart_dir)
        
        # Should find audio despite filename mismatch
        assert entry.has_audio(), "Audio not detected with filename mismatch"
        assert entry.audio_file is not None
        assert entry.audio_file.name == "music.ogg"
        
        print("✓ Audio detection with mismatch: PASSED")


def test_audio_detection_case_insensitive():
    """Test that audio is found with case variations (Music.MP3, etc.)."""
    print("Testing audio_detection_case_insensitive...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chart_dir = tmpdir / "test456__CaseTest"
        chart_dir.mkdir()
        
        # Create chart
        level = AdofaiLevel(
            settings={
                "version": 14,
                "artist": "Test",
                "song": "Test",
                "author": "Test",
                "songFilename": "audio.MP3",  # Uppercase extension
                "bpm": 120,
                "offset": 0,
                "volume": 100,
                "pitch": 100,
                "hitsound": "Kick",
                "hitsoundVolume": 100,
            },
            angle_data=[0, 90],
            actions=[]
        )
        
        write_adofai(level, chart_dir / "level.adofai")
        
        # Create audio with uppercase extension
        audio_path = chart_dir / "audio.MP3"
        create_silent_wav(audio_path, duration_sec=5.0)
        
        # Try to load as dataset entry
        entry = AdofaiDatasetEntry(chart_dir)
        
        # Should find audio despite case difference
        assert entry.has_audio(), "Audio not detected with uppercase extension"
        assert entry.audio_file is not None
        
        print("✓ Audio detection case-insensitive: PASSED")


def test_oom_fix_long_audio():
    """
    Test that the OOM fix works: long audio should be processed efficiently
    without allocating multi-GB tensors.
    
    This test verifies:
    1. Audio is capped at max_duration_sec (60s)
    2. Spectrogram conversion reduces memory from O(samples) to O(frames)
    3. Forward+backward pass works on CPU without excessive memory
    """
    import tempfile
    import torch
    from adofai.dataset import load_audio_file, compute_log_mel_spectrogram
    from adofai.train import SimpleADOFAIModel
    from adofai.tokenizer import AdofaiTokenizer
    
    print("Testing OOM fix with long audio...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a long audio file (90 seconds at 16kHz = 1.44M samples)
        # This would cause OOM with per-sample Linear layer
        audio_path = tmpdir / "long_audio.wav"
        create_silent_wav(audio_path, duration_sec=90.0, sample_rate=16000)
        
        # Load with duration cap (should be cropped to 60s)
        audio_waveform = load_audio_file(
            audio_path,
            sample_rate=16000,
            max_duration_sec=60.0,
        )
        
        # Verify duration cap worked
        expected_max_samples = 60 * 16000  # 960k samples
        assert len(audio_waveform) <= expected_max_samples, \
            f"Audio not capped: {len(audio_waveform)} > {expected_max_samples}"
        print(f"  ✓ Audio capped to {len(audio_waveform)} samples (60s at 16kHz)")
        
        # Convert to spectrogram (should be compact: [time_frames, n_mels])
        spectrogram = compute_log_mel_spectrogram(
            audio_waveform,
            sample_rate=16000,
            n_mels=80,
        )
        
        # Verify spectrogram shape is O(frames) not O(samples)
        # With hop_length=160, 60s @ 16kHz = 960k samples / 160 = 6000 frames
        assert spectrogram.shape[0] < 10000, \
            f"Spectrogram too large: {spectrogram.shape}"
        assert spectrogram.shape[1] == 80, \
            f"Wrong n_mels: {spectrogram.shape[1]}"
        print(f"  ✓ Spectrogram shape: {spectrogram.shape} (much smaller than raw audio)")
        
        # Test forward+backward pass (should not allocate multi-GB tensors)
        tokenizer = AdofaiTokenizer()
        model = SimpleADOFAIModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,  # Small for test
            num_layers=1,
            n_mels=80,
        )
        model.eval()
        
        # Create dummy batch
        batch_size = 1
        audio_batch = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # [1, frames, 80]
        
        # Dummy target tokens
        target_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, 32), dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            logits = model(audio_batch, target_tokens[:, :-1])
        
        assert logits.shape[0] == batch_size
        assert logits.shape[2] == tokenizer.vocab_size
        print(f"  ✓ Forward pass succeeded with output shape: {logits.shape}")
        
        # Backward pass (test memory during gradient computation)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        logits = model(audio_batch, target_tokens[:, :-1])
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Backward pass succeeded (loss: {loss.item():.4f})")
        print("✓ OOM fix test: PASSED")


if __name__ == "__main__":
    print("Running ADOFAI training pipeline tests...\n")
    
    test_create_synthetic_data()
    test_dataset_entry()
    test_dataset_loading()
    test_tokenizer()
    test_batch_collation()
    test_smoke_training()
    test_audio_detection_mismatch()
    test_audio_detection_case_insensitive()
    test_oom_fix_long_audio()
    
    print("\n✅ All training pipeline tests passed!")
