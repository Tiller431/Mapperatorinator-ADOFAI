"""
DEPRECATED STUB: SimpleADOFAIModel LSTM training path.

This file is kept for reference only and should NOT be used for production training.

Production training path: python osuT5/train.py -cn adofai_v31
    - Uses Whisper encoder-decoder (Tiger14n/ropewhisper-small)
    - Full event vocabulary (SetSpeed, Twirl, camera, VFX, difficulty)
    - Lossless augmentation (rotate, reflect, pitch, rate)
    - v31 config: muon optimizer, bf16, 65536 steps
    - See configs/train/adofai_v31.yaml

This SimpleADOFAIModel is an LSTM stub and does NOT match the v31 architecture.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .dataset import AdofaiDataset, collate_adofai_batch
from .tokenizer import AdofaiTokenizer


class SimpleADOFAIModel(nn.Module):
    """
    Minimal LSTM-based model for ADOFAI generation (smoke test).
    
    For production, use the full Whisper-based architecture from osuT5.
    This is just for testing the training pipeline.
    
    NOTE: This model expects log-mel spectrogram input [batch, time_frames, n_mels]
    instead of raw audio waveforms to avoid OOM on long songs.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        n_mels: int = 80,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_mels = n_mels
        
        # Audio encoder: takes spectrogram features [time_frames, n_mels]
        # Uses a small conv + linear to encode efficiently
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Pooling to reduce time dimension further
        self.audio_pool = nn.AdaptiveAvgPool1d(32)  # Reduce to fixed 32 frames
        
        # Decoder: LSTM + output head
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, audio, target_tokens=None):
        """
        Forward pass.
        
        Args:
            audio: [batch, time_frames, n_mels] log-mel spectrogram
            target_tokens: [batch, seq_len] target token IDs (for teacher forcing)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = audio.shape[0]
        
        # Encode audio spectrogram
        # audio: [batch, time_frames, n_mels]
        # Conv expects [batch, channels, time], so transpose
        audio_t = audio.transpose(1, 2)  # [batch, n_mels, time_frames]
        audio_encoded = self.audio_encoder(audio_t)  # [batch, hidden, time_frames]
        
        # Pool to fixed size to keep memory bounded
        audio_pooled = self.audio_pool(audio_encoded)  # [batch, hidden, 32]
        
        # Transpose back and mean pool to single context vector
        audio_features = audio_pooled.transpose(1, 2)  # [batch, 32, hidden]
        audio_context = audio_features.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        if target_tokens is not None:
            # Teacher forcing during training
            token_embeds = self.decoder_embedding(target_tokens)  # [batch, seq_len, hidden]
            
            # Concatenate audio context
            decoder_input = torch.cat([audio_context, token_embeds], dim=1)  # [batch, seq_len+1, hidden]
            
            # Decode
            decoder_output, _ = self.decoder_lstm(decoder_input)
            logits = self.output_head(decoder_output[:, :-1, :])  # Remove last position
            
            return logits
        else:
            # Autoregressive generation for inference
            return audio_context  # Return context for use by generate() method
    
    def generate(
        self,
        audio,
        max_length: int = 512,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """
        Generate tokens autoregressively from audio.
        
        Args:
            audio: [batch, time_frames, n_mels] log-mel spectrogram
            max_length: Maximum sequence length to generate
            bos_token_id: Begin-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            temperature: Sampling temperature (1.0 = no change, <1 = more confident, >1 = more random)
            top_k: Only sample from top k most likely tokens (0 = no filtering)
        
        Returns:
            generated_tokens: [batch, seq_len] generated token IDs
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        # Get audio context
        audio_context = self.forward(audio, target_tokens=None)  # [batch, 1, hidden]
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # LSTM hidden state
        lstm_hidden = None
        
        for step in range(max_length):
            # Get embeddings for current sequence
            if step == 0:
                # First step: use audio context
                decoder_input = audio_context  # [batch, 1, hidden]
            else:
                # Subsequent steps: embed last generated token
                last_token = generated[:, -1:]  # [batch, 1]
                token_embed = self.decoder_embedding(last_token)  # [batch, 1, hidden]
                decoder_input = token_embed
            
            # LSTM forward
            decoder_output, lstm_hidden = self.decoder_lstm(decoder_input, lstm_hidden)
            
            # Get logits for next token
            logits = self.output_head(decoder_output[:, -1, :])  # [batch, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: AdofaiTokenizer,
    device: str = 'cpu',
    max_seq_len: int = 512,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        # Move audio to device
        audio = batch['audio'].to(device)
        
        # Tokenize events
        batch_tokens = []
        for events in batch['events']:
            tokens = tokenizer.events_to_tokens(events)
            # Truncate if too long
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            batch_tokens.append(tokens)
        
        # Pad to max length in batch
        max_len = max(len(tokens) for tokens in batch_tokens)
        padded_tokens = []
        for tokens in batch_tokens:
            padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            padded_tokens.append(padded)
        
        target_tokens = torch.tensor(padded_tokens, dtype=torch.long, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(audio, target_tokens[:, :-1])  # Shift right for autoregression
        
        # Compute loss (cross-entropy, ignore padding)
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            target_tokens[:, 1:].reshape(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train ADOFAI generation model")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing ADOFAI chart folders')
    parser.add_argument('--index_json', type=str, default=None,
                        help='Optional JSON index of charts')
    parser.add_argument('--output_dir', type=str, default='adofai_checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default 2 for T4 GPU safety)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples (for smoke testing)')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke mode: tiny model, few steps')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda/mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Smoke mode overrides
    if args.smoke:
        print("🔥 SMOKE MODE: Using tiny model and limited data")
        args.max_samples = 5
        args.batch_size = 2
        args.epochs = 2
        hidden_size = 64
        num_layers = 1
    else:
        hidden_size = 256
        num_layers = 2
        # Use batch_size=1 for full training on T4 GPU to avoid OOM
        if args.device == 'cuda' and args.batch_size > 2:
            print(f"⚠️  Batch size {args.batch_size} may cause OOM on T4 GPU")
            print("    Recommended: --batch_size 1 or 2 for full training")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AdofaiTokenizer()
    tokenizer.save(output_dir / 'tokenizer.pkl')
    
    # Create datasets
    print(f"Loading training data from {args.data_dir}...")
    train_dataset = AdofaiDataset(
        data_dir=args.data_dir,
        index_json=args.index_json,
        split='train',
        max_samples=args.max_samples,
    )
    
    val_dataset = AdofaiDataset(
        data_dir=args.data_dir,
        index_json=args.index_json,
        split='val',
        max_samples=args.max_samples // 2 if args.max_samples else None,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_adofai_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_adofai_batch,
    )
    
    # Initialize model
    print(f"Initializing model (vocab_size={tokenizer.vocab_size})...")
    model = SimpleADOFAIModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, tokenizer,
            device=args.device
        )
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch{epoch+1}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'model_config': {
                'vocab_size': tokenizer.vocab_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'n_mels': 80,
            },
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    print(f"\n✅ Training complete! Checkpoints saved to {output_dir}")
    print(f"\nNote: This is a smoke test model. For production quality:")
    print("  1. Use full Whisper encoder from osuT5")
    print("  2. Train on full ADOFAI dataset (100+ charts)")
    print("  3. Integrate with osuT5 training pipeline")
    print("\nT4 GPU Settings (to avoid OOM):")
    print("  - Audio capped at 60s (center crop)")
    print("  - Log-mel spectrogram features (80 mels)")
    print("  - Batch size 1-2 recommended")
    print("  - Max sequence length 512 tokens")


if __name__ == '__main__':
    main()
