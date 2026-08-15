"""
ADOFAI inference CLI - generate charts from audio using trained checkpoints.

Usage:
    python -m adofai.inference_cli \
        --audio_path path/to/audio.mp3 \
        --checkpoint path/to/checkpoint_epoch50.pt \
        --tokenizer path/to/tokenizer.pkl \
        --output_path output/ \
        --title "Song Title" \
        --artist "Artist Name"
"""

import argparse
from pathlib import Path
import torch

from .train import SimpleADOFAIModel
from .tokenizer import AdofaiTokenizer
from .dataset import load_audio_file, compute_log_mel_spectrogram
from .inference import export_adofai_from_events
from .converter import AdofaiConverter
from .event import AdofaiEventType


def load_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """
    Load checkpoint from training.
    
    Handles both formats:
    - Full checkpoint dict with 'model_state_dict', 'epoch', 'model_config'
    - Direct state_dict (assumes default config)
    
    Returns:
        state_dict, metadata, model_config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format from train.py
        return (
            checkpoint['model_state_dict'],
            {
                'epoch': checkpoint.get('epoch', None),
                'train_loss': checkpoint.get('train_loss', None),
            },
            checkpoint.get('model_config', {
                'hidden_size': 256,
                'num_layers': 2,
                'n_mels': 80,
            })
        )
    else:
        # Direct state_dict - assume default config
        return checkpoint, {}, {
            'hidden_size': 256,
            'num_layers': 2,
            'n_mels': 80,
        }


def generate_adofai_from_audio(
    audio_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    title: str = "Generated Song",
    artist: str = "Unknown Artist",
    bpm: float = 120.0,
    offset: int = 0,
    device: str = 'cpu',
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """
    Generate ADOFAI chart from audio file using trained checkpoint.
    
    Args:
        audio_path: Path to audio file
        checkpoint_path: Path to model checkpoint (.pt file)
        tokenizer_path: Path to tokenizer (.pkl file)
        output_path: Directory to save output .adofai
        title: Song title
        artist: Artist name
        bpm: Base BPM (can be overridden by model)
        offset: Timing offset in ms
        device: Device to run inference on ('cpu', 'cuda', 'mps')
        max_length: Maximum sequence length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        
    Returns:
        Path to generated .adofai file
    """
    print(f"🎵 Loading audio from {audio_path}...")
    
    # Load and preprocess audio (same as training)
    audio_waveform = load_audio_file(
        audio_path,
        sample_rate=16000,
        max_duration_sec=60.0,
    )
    
    # Convert to log-mel spectrogram
    audio_features = compute_log_mel_spectrogram(
        audio_waveform,
        sample_rate=16000,
        n_mels=80,
    )
    
    print(f"  Audio shape: {audio_waveform.shape} samples")
    print(f"  Spectrogram shape: {audio_features.shape}")
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer from {tokenizer_path}...")
    tokenizer = AdofaiTokenizer.load(tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Load model
    print(f"\n🤖 Loading model from {checkpoint_path}...")
    state_dict, metadata, model_config = load_checkpoint(checkpoint_path, device=device)
    
    if metadata.get('epoch') is not None:
        print(f"  Checkpoint from epoch {metadata['epoch'] + 1}")
    if metadata.get('train_loss') is not None:
        print(f"  Training loss: {metadata['train_loss']:.4f}")
    
    # Initialize model with config from checkpoint
    model = SimpleADOFAIModel(
        vocab_size=model_config.get('vocab_size', tokenizer.vocab_size),
        hidden_size=model_config.get('hidden_size', 256),
        num_layers=model_config.get('num_layers', 2),
        n_mels=model_config.get('n_mels', 80),
    )
    
    print(f"  Model config: hidden_size={model_config.get('hidden_size', 256)}, "
          f"num_layers={model_config.get('num_layers', 2)}, n_mels={model_config.get('n_mels', 80)}")
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    
    # Prepare audio batch
    audio_batch = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0)  # [1, frames, mels]
    audio_batch = audio_batch.to(device)
    
    # Generate tokens
    print(f"\n✨ Generating chart (max_length={max_length}, temperature={temperature}, top_k={top_k})...")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            audio_batch,
            max_length=max_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Convert to list and remove batch dimension
    token_ids = generated_tokens[0].cpu().tolist()
    print(f"  Generated {len(token_ids)} tokens")
    
    # Decode tokens to events
    print(f"\n🎮 Converting tokens to ADOFAI events...")
    events = []
    event_times = []
    current_time = 0.0
    
    for token_id in token_ids:
        # Skip special tokens
        if token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            continue
        
        # Decode token
        event = tokenizer.token_to_event(token_id)
        
        if event is None:
            continue
        
        # Track time
        if event.type == AdofaiEventType.TIME_SHIFT:
            current_time += event.value
        
        events.append(event)
        event_times.append(current_time)
    
    print(f"  Decoded {len(events)} events")
    
    # Export to .adofai
    print(f"\n💾 Exporting to {output_path}...")
    
    audio_basename = audio_path.name
    
    output_file = export_adofai_from_events(
        events=events,
        event_times=event_times,
        output_path=output_path,
        audio_filename=audio_basename,
        title=title,
        artist=artist,
        creator="Mapperatorinator ADOFAI",
    )
    
    print(f"\n✅ Generated chart saved to: {output_file}")
    print(f"   Copy your audio file ({audio_basename}) to the same directory to play it!")
    
    return output_file


def main():
    raise SystemExit(
        "adofai/inference_cli.py is not the production path. "
        "Train with: python osuT5/train.py -cn adofai_v31"
    )
    parser = argparse.ArgumentParser(
        description="Generate ADOFAI charts from audio using trained model"
    )
    
    # Required arguments
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer file (.pkl file)')
    parser.add_argument('--output_path', type=str, default='output',
                        help='Output directory for .adofai file')
    
    # Optional metadata
    parser.add_argument('--title', type=str, default='Generated Song',
                        help='Song title')
    parser.add_argument('--artist', type=str, default='Unknown Artist',
                        help='Artist name')
    parser.add_argument('--bpm', type=float, default=120.0,
                        help='Base BPM (can be overridden by model)')
    parser.add_argument('--offset', type=int, default=0,
                        help='Timing offset in milliseconds')
    
    # Generation parameters
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run inference on')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling (0 = no filtering)')
    
    args = parser.parse_args()
    
    # Convert paths
    audio_path = Path(args.audio_path)
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output_path)
    
    # Validate inputs
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Generate chart
    generate_adofai_from_audio(
        audio_path=audio_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        title=args.title,
        artist=args.artist,
        bpm=args.bpm,
        offset=args.offset,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
