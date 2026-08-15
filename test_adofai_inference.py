"""
Test ADOFAI inference pipeline.
"""

import tempfile
from pathlib import Path
import torch
import numpy as np

from adofai.train import SimpleADOFAIModel
from adofai.tokenizer import AdofaiTokenizer
from adofai.inference_cli import generate_adofai_from_audio


def create_test_wav(path: Path, duration_sec: float = 5.0, sample_rate: int = 16000):
    """Create a simple test WAV file."""
    from scipy.io import wavfile
    
    # Generate a simple sine wave
    t = np.linspace(0, duration_sec, int(duration_sec * sample_rate))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    audio = (audio * 32767).astype(np.int16)
    
    wavfile.write(path, sample_rate, audio)


def test_model_generate():
    """Test that SimpleADOFAIModel.generate() works without NotImplementedError."""
    print("Testing model.generate()...")
    
    # Create tokenizer
    tokenizer = AdofaiTokenizer()
    
    # Create model
    model = SimpleADOFAIModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=1,
        n_mels=80,
    )
    model.eval()
    
    # Create dummy spectrogram
    audio = torch.randn(1, 100, 80)  # [batch=1, time_frames=100, n_mels=80]
    
    # Generate tokens (should not raise NotImplementedError)
    with torch.no_grad():
        generated = model.generate(
            audio,
            max_length=20,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    assert generated.shape[0] == 1, "Batch size mismatch"
    assert generated.shape[1] > 1, "Should generate tokens"
    
    print(f"  ✓ Generated {generated.shape[1]} tokens")
    print("✓ model.generate() test: PASSED")


def test_tokenizer_roundtrip():
    """Test tokenizer can decode tokens back to events."""
    print("\nTesting tokenizer decode...")
    
    from adofai.event import AdofaiEvent, AdofaiEventType
    
    tokenizer = AdofaiTokenizer()
    
    # Create some test events
    events = [
        AdofaiEvent(AdofaiEventType.TIME_SHIFT, 100),
        AdofaiEvent(AdofaiEventType.TILE_ANGLE, 90),
        AdofaiEvent(AdofaiEventType.TIME_SHIFT, 200),
        AdofaiEvent(AdofaiEventType.TILE_ANGLE, 180),
        AdofaiEvent(AdofaiEventType.TWIRL, 0),
    ]
    
    # Encode
    token_ids = tokenizer.events_to_tokens(events)
    print(f"  Encoded {len(events)} events to {len(token_ids)} tokens")
    
    # Decode
    decoded_events = []
    for token_id in token_ids:
        event = tokenizer.token_to_event(token_id)
        if event is not None:
            decoded_events.append(event)
    
    print(f"  Decoded to {len(decoded_events)} events")
    assert len(decoded_events) >= len(events) - 1, "Should recover most events"
    
    print("✓ tokenizer decode test: PASSED")


def test_inference_pipeline():
    """Test full inference pipeline: audio → checkpoint → .adofai."""
    print("\nTesting full inference pipeline...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        audio_path = tmpdir / "test_audio.wav"
        create_test_wav(audio_path, duration_sec=5.0)
        print(f"  Created test audio: {audio_path}")
        
        # Create tokenizer
        tokenizer = AdofaiTokenizer()
        tokenizer_path = tmpdir / "tokenizer.pkl"
        tokenizer.save(tokenizer_path)
        print(f"  Saved tokenizer: {tokenizer_path}")
        
        # Create and save model
        model = SimpleADOFAIModel(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_layers=1,
            n_mels=80,
        )
        
        checkpoint_path = tmpdir / "checkpoint_epoch1.pt"
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'train_loss': 1.0,
            'model_config': {
                'vocab_size': tokenizer.vocab_size,
                'hidden_size': 64,
                'num_layers': 1,
                'n_mels': 80,
            },
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Run inference
        output_path = tmpdir / "output"
        output_path.mkdir()
        
        result_file = generate_adofai_from_audio(
            audio_path=audio_path,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            output_path=output_path,
            title="Test Chart",
            artist="Test Artist",
            device='cpu',
            max_length=50,
        )
        
        # Verify output
        assert result_file.exists(), "Output .adofai file not created"
        assert result_file.suffix == '.adofai', "Wrong file extension"
        
        # Try to parse it
        from adofai.parser import parse_adofai
        level = parse_adofai(result_file)
        
        assert level.settings['song'] == "Test Chart"
        assert level.settings['artist'] == "Test Artist"
        assert level.settings['songFilename'] == "test_audio.wav"
        assert len(level.angle_data) > 0, "Should have some tiles"
        
        print(f"  Generated chart with {len(level.angle_data)} tiles")
        print(f"  Output file: {result_file}")
        print("✓ full inference pipeline test: PASSED")


if __name__ == "__main__":
    print("Running ADOFAI inference tests...\n")
    
    test_model_generate()
    test_tokenizer_roundtrip()
    test_inference_pipeline()
    
    print("\n✅ All inference tests passed!")
