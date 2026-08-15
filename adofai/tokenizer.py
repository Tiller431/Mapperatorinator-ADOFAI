"""
Tokenizer for ADOFAI events.

Converts ADOFAI events to/from token IDs for model training.
"""

from __future__ import annotations
from typing import Optional
import pickle

from .event import AdofaiEvent, AdofaiEventType


class AdofaiTokenizer:
    """
    Tokenizer for ADOFAI event sequences.
    
    Builds a vocabulary from ADOFAI event types and values:
    - Special tokens: <pad>, <sos>, <eos>, <unk>
    - Event types: each AdofaiEventType gets tokens
    - Angle values: 0-359 (360 tokens)
    - Midspin: special token for 999
    - Speed/BPM values: quantized range
    - Other numeric values: quantized ranges
    """
    
    def __init__(
        self,
        max_bpm: int = 300,
        bpm_quantize: int = 1,
        max_time_shift: int = 10000,  # 10 seconds
        time_quantize: int = 10,  # 10ms precision
    ):
        """
        Initialize tokenizer with vocabulary ranges.
        
        Args:
            max_bpm: Maximum BPM value to support
            bpm_quantize: BPM quantization step
            max_time_shift: Maximum time shift in ms
            time_quantize: Time quantization in ms
        """
        self.max_bpm = max_bpm
        self.bpm_quantize = bpm_quantize
        self.max_time_shift = max_time_shift
        self.time_quantize = time_quantize
        
        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Build token vocabulary."""
        idx = 0
        
        # Special tokens
        for token in ['<pad>', '<sos>', '<eos>', '<unk>']:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        self.pad_token_id = self.token_to_id['<pad>']
        self.sos_token_id = self.token_to_id['<sos>']
        self.bos_token_id = self.sos_token_id  # Alias for consistency
        self.eos_token_id = self.token_to_id['<eos>']
        self.unk_token_id = self.token_to_id['<unk>']
        
        # Time shift tokens (quantized)
        for time_ms in range(0, self.max_time_shift, self.time_quantize):
            token = f"time{time_ms}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Angle tokens (0-359)
        for angle in range(360):
            token = f"angle{angle}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Midspin token
        self.token_to_id['midspin'] = idx
        self.id_to_token[idx] = 'midspin'
        idx += 1
        
        # BPM tokens (quantized)
        for bpm in range(0, self.max_bpm + 1, self.bpm_quantize):
            token = f"bpm{bpm}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Speed multiplier tokens (0.1 to 5.0 in 0.1 steps)
        for mult_int in range(1, 51):  # 0.1 to 5.0
            mult = mult_int / 10.0
            token = f"speedmult{mult:.1f}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Special event tokens
        for event_token in ['twirl', 'pause', 'hold', 'multiplanet']:
            self.token_to_id[event_token] = idx
            self.id_to_token[idx] = event_token
            idx += 1
        
        # Duration tokens (for pause/hold, 0.1 to 10.0 beats in 0.1 steps)
        for dur_int in range(1, 101):
            dur = dur_int / 10.0
            token = f"dur{dur:.1f}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Offset tokens (-1000 to 1000 ms in 10ms steps)
        for offset_ms in range(-1000, 1001, 10):
            token = f"offset{offset_ms}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        self.vocab_size = idx
        print(f"Built ADOFAI tokenizer with vocab size: {self.vocab_size}")
    
    def quantize_time(self, time_ms: float) -> int:
        """Quantize time to nearest time_quantize step."""
        return int(round(time_ms / self.time_quantize) * self.time_quantize)
    
    def quantize_bpm(self, bpm: float) -> int:
        """Quantize BPM to nearest bpm_quantize step."""
        return int(round(bpm / self.bpm_quantize) * self.bpm_quantize)
    
    def event_to_tokens(self, event: AdofaiEvent) -> list[int]:
        """
        Convert single event to token IDs.
        
        Returns list of tokens (some events may produce multiple tokens).
        """
        tokens = []
        
        if event.type == AdofaiEventType.TIME_SHIFT:
            time_ms = self.quantize_time(event.value)
            time_ms = min(time_ms, self.max_time_shift - self.time_quantize)
            token = f"time{time_ms}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.TILE_ANGLE:
            angle = int(event.value) % 360
            token = f"angle{angle}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.MIDSPIN:
            tokens.append(self.token_to_id['midspin'])
        
        elif event.type == AdofaiEventType.BPM:
            bpm = self.quantize_bpm(event.value)
            bpm = min(bpm, self.max_bpm)
            token = f"bpm{bpm}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.SET_SPEED_BPM:
            bpm = self.quantize_bpm(event.value)
            bpm = min(bpm, self.max_bpm)
            token = f"bpm{bpm}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.SET_SPEED_MULT:
            mult = round(event.value * 10) / 10  # Round to 0.1
            mult = max(0.1, min(mult, 5.0))
            token = f"speedmult{mult:.1f}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.TWIRL:
            tokens.append(self.token_to_id['twirl'])
        
        elif event.type == AdofaiEventType.PAUSE:
            tokens.append(self.token_to_id['pause'])
            dur = round(event.value * 10) / 10
            dur = max(0.1, min(dur, 10.0))
            token = f"dur{dur:.1f}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.HOLD:
            tokens.append(self.token_to_id['hold'])
            dur = round(event.value * 10) / 10
            dur = max(0.1, min(dur, 10.0))
            token = f"dur{dur:.1f}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        elif event.type == AdofaiEventType.MULTI_PLANET:
            tokens.append(self.token_to_id['multiplanet'])
        
        elif event.type == AdofaiEventType.OFFSET:
            offset_ms = int(round(event.value / 10) * 10)
            offset_ms = max(-1000, min(offset_ms, 1000))
            token = f"offset{offset_ms}"
            tokens.append(self.token_to_id.get(token, self.unk_token_id))
        
        else:
            # Unknown event type
            tokens.append(self.unk_token_id)
        
        return tokens
    
    def events_to_tokens(self, events: list[AdofaiEvent]) -> list[int]:
        """Convert event sequence to token IDs."""
        tokens = [self.sos_token_id]
        
        for event in events:
            tokens.extend(self.event_to_tokens(event))
        
        tokens.append(self.eos_token_id)
        return tokens
    
    def token_to_event(self, token_id: int) -> Optional[AdofaiEvent]:
        """
        Convert single token ID back to event.
        
        Returns None for special tokens or unrecognized tokens.
        """
        if token_id not in self.id_to_token:
            return None
        
        token_str = self.id_to_token[token_id]
        
        # Special tokens
        if token_str in ['<pad>', '<sos>', '<eos>', '<unk>']:
            return None
        
        # Time shift
        if token_str.startswith('time'):
            time_ms = int(token_str[4:])
            return AdofaiEvent(AdofaiEventType.TIME_SHIFT, time_ms)
        
        # Angle
        if token_str.startswith('angle'):
            angle = int(token_str[5:])
            return AdofaiEvent(AdofaiEventType.TILE_ANGLE, angle)
        
        # Midspin
        if token_str == 'midspin':
            return AdofaiEvent(AdofaiEventType.MIDSPIN, 999)
        
        # BPM
        if token_str.startswith('bpm'):
            bpm = int(token_str[3:])
            return AdofaiEvent(AdofaiEventType.SET_SPEED_BPM, bpm)
        
        # Speed multiplier
        if token_str.startswith('speedmult'):
            mult = float(token_str[9:])
            return AdofaiEvent(AdofaiEventType.SET_SPEED_MULT, mult)
        
        # Twirl
        if token_str == 'twirl':
            return AdofaiEvent(AdofaiEventType.TWIRL, 0)
        
        # Pause (duration follows)
        if token_str == 'pause':
            return AdofaiEvent(AdofaiEventType.PAUSE, 0.0)
        
        # Hold (duration follows)
        if token_str == 'hold':
            return AdofaiEvent(AdofaiEventType.HOLD, 0.0)
        
        # Duration
        if token_str.startswith('dur'):
            # This is a duration value that modifies previous pause/hold
            # Return as a special marker that caller can handle
            return None
        
        # MultiPlanet
        if token_str == 'multiplanet':
            return AdofaiEvent(AdofaiEventType.MULTI_PLANET, 2)
        
        # Offset
        if token_str.startswith('offset'):
            offset = int(token_str[6:])
            return AdofaiEvent(AdofaiEventType.OFFSET, offset)
        
        # Unknown
        return None
    
    def save(self, path: str):
        """Save tokenizer to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'max_bpm': self.max_bpm,
                'bpm_quantize': self.bpm_quantize,
                'max_time_shift': self.max_time_shift,
                'time_quantize': self.time_quantize,
                'vocab_size': self.vocab_size,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'AdofaiTokenizer':
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            max_bpm=data['max_bpm'],
            bpm_quantize=data['bpm_quantize'],
            max_time_shift=data['max_time_shift'],
            time_quantize=data['time_quantize'],
        )
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = data['id_to_token']
        tokenizer.vocab_size = data['vocab_size']
        
        return tokenizer
