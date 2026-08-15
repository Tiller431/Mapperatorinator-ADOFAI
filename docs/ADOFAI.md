# ADOFAI Format Support

This document describes the ADOFAI (A Dance of Fire and Ice) format implementation in Mapperatorinator-ADOFAI, including file format details, event representation, and training requirements.

## Overview

ADOFAI is a one-button rhythm game where players control two orbiting spheres navigating a path of tiles. Charts are defined by:
- **Tile angles** — the direction each tile points (0-359 degrees)
- **Timing events** — speed changes (SetSpeed), pauses, holds
- **Gameplay modifiers** — twirls (direction changes), MultiPlanet

This project adapts the Mapperatorinator architecture (originally for osu!) to generate ADOFAI charts from audio spectrograms.

## ADOFAI File Format (`.adofai`)

ADOFAI charts are stored as JSON-like files with the following structure:

```json
{
    "settings": {
        "version": 14,
        "artist": "Artist Name",
        "song": "Song Title",
        "author": "Chart Creator",
        "bpm": 120,
        "offset": 0,
        "songFilename": "audio.ogg",
        ...
    },
    "angleData": [0, 0, 90, 180, 270, 45, 999, ...],
    "actions": [
        {
            "floor": 1,
            "eventType": "SetSpeed",
            "speedType": "Bpm",
            "beatsPerMinute": 120,
            "bpmMultiplier": 1.0
        },
        {
            "floor": 5,
            "eventType": "Twirl"
        },
        ...
    ],
    "decorations": [...]
}
```

### Key Fields

- **`settings`** — Global metadata (BPM, offset, audio file, artist, title, etc.)
- **`angleData`** — Array of tile angles:
  - `0-359`: Angle in degrees (0 = right, 90 = up, 180 = left, 270 = down)
  - `999`: Special midspin tile (planet continues rotating without stopping)
- **`actions`** — Events attached to specific floors (tiles):
  - `SetSpeed`: Change BPM or apply multiplier
  - `Twirl`: Reverse rotation direction
  - `Pause`: Pause movement for duration
  - `Hold`: Hold input for duration
  - `MultiPlanet`: Multiple orbiting planets
  - Camera/VFX events (not yet supported in v1)
- **`decorations`** — Visual decorations (not yet supported in v1)

### Format Quirks

ADOFAI files are *almost* JSON but with quirks:
- **UTF-8 BOM**: Steam Workshop files often start with a UTF-8 Byte Order Mark. Our parser uses `encoding='utf-8-sig'` to automatically strip it.
- **Trailing commas** are common and must be tolerated
- **Legacy pathData**: Older files use `pathData` (string like `"RRULDR!"`) instead of `angleData`
  - Conversion table: `R=0°, U=90°, L=180°, D=270°, !=999 (midspin)`, etc.

Our parser handles both formats and always writes modern `angleData`.

## Event Representation

The `adofai/event.py` module defines intermediate events for model training/inference:

### Event Types (v1 Scope)

**Timing:**
- `TIME_SHIFT` — Timestamp in milliseconds

**Tiles:**
- `TILE_ANGLE` — Angle in degrees (0-359)
- `MIDSPIN` — Special 999 tile

**Speed:**
- `SET_SPEED_BPM` — Change BPM
- `SET_SPEED_MULT` — Apply speed multiplier
- `PAUSE` — Pause duration
- `HOLD` — Hold duration

**Gameplay:**
- `TWIRL` — Direction change
- `MULTI_PLANET` — Number of planets

**Metadata (conditioning):**
- `BPM`, `OFFSET`, `DIFFICULTY`, `SONG_LENGTH`

### Design Decisions

1. **Integer angles (0-359)** — No quantization to preserve expressiveness
2. **Midspin as distinct event** — `999` is a special tile type, not an angle
3. **Floor-to-time conversion** — Actions use floor indices; we convert to/from timestamps using BPM and angle rotation math
4. **v1 excludes camera/VFX** — Deferred to focus on playable rhythm mechanics

## Timing Calculations

ADOFAI timing is complex because tiles are reached based on *angle rotation*, not fixed time intervals:

1. **Rotation speed**: Planet rotates at 180° per beat
2. **Angle difference**: Between consecutive tiles determines time
3. **Twirl direction**: Affects whether rotation is clockwise or counter-clockwise
4. **BPM changes**: `SetSpeed` actions modify rotation speed mid-chart

Formula:
```
time_delta = (angle_difference / 180) * (60000 / current_bpm)
```

The `AdofaiConverter` class handles this calculation when converting between `.adofai` structure and event sequences.

## Training with Whisper Encoder-Decoder

ADOFAI training uses the full osuT5 Whisper infrastructure (same as osu! Mapperatorinator).

### Model Architecture

**Production Model:**
- **Encoder-Decoder:** Tiger14n/ropewhisper-small (~219M parameters, 12/12/768)
- **Spectrogram:** torchaudio Mel (16kHz, hop 128, n_fft 1024, 80 mels, log scale)
- **Context:** src 4096 frames (~32.8s), tgt 8192 tokens (staged: timing → notes/actions)
- **Loss:** Token cross-entropy, rhythm_weight 1.0
- **Optimizer:** Muon (lr 1e-2), 65536 steps, bf16, compile
- **Conditioning:** Difficulty ON (encoder RBF embed + prefix token), NO style/descriptor/year tags

**Smoke Model (testing only):**
- **Encoder-Decoder:** Tiger14n/ropewhisper-tiny (~39M parameters)
- **Context:** src 1024 frames (~8.2s), tgt 2048 tokens
- **Optimizer:** AdamW (lr 1e-4), 100 steps, no bf16, no compile
- **Purpose:** Verify training loop, vocab emission, augmentation — NOT for quality generation

### Dataset Layout

Training expects Workshop-style folders:

```
<data_root>/
  <workshopId>__<chartName>/
    level.adofai
    audio.ogg  # or .mp3, .wav, .flac, .m4a, .aac
  
  another_id__AnotherChart/
    level.adofai
    song.mp3
```

**Lossless Augmentation (continuous uniform, independent):**

Each chart yields **ONE augmented variant per epoch** with random sampling. Transforms are **INDEPENDENT** (rotate AND reflect MAY both apply, not XOR).

- **Rotation:** With probability p_rotate (default 1.0), sample R ~ Uniform[0, 360)
  - Apply: a = (a + R) % 360 for non-999 angles; 999 unchanged
  - Twirls unchanged; camera/track positions and rotations also rotated
  - angleOffset unchanged

- **Reflection:** With probability p_reflect (default 0.5), pick one axis from proven family
  - Axes: X-flip (−a % 360), Y-flip (180−a), y=x (90−a), y=−x (270−a)
  - Add floor-0 Twirl (toggle if already exists)
  - 999 unchanged

- **Matched-rate:** With probability p_rate (default 0.5), sample r ~ Uniform[0.85, 1.25]
  - Scales: BPM × r, SetSpeed BPM × r, offset ms / r
  - Audio duration → duration / r
  - Leaves unchanged: multipliers, Pause/Hold/camera durations (in beats), angleOffset

- **Same-duration pitch:** With probability p_pitch (default 0.5), sample settings.pitch ~ Uniform[80, 120]
  - Waveform pitch-shift without duration change
  - Chart events untouched; 100 = no pitch change

**Transforms are independent:** Each is sampled separately; no discrete steps, no XOR locks, no cartesian product. Chart sees different random augmentation each epoch.

**Expected effective dataset size:** ~126 base charts with stochastic augmentation (infinite variants due to continuous sampling).

**Dataset:** [Google Drive top-100 archive](https://drive.google.com/drive/folders/1lATJxQI8P3uLsRtiC7ay5u3SrFhH1cfd) (`adofai-top100.tar.gz`, ~126 charts)

### Event Coverage

**Full vocab (all in this train):**
- **Timing:** BPM, offset, TIME_SHIFT, SetSpeed (BPM + multiplier), Pause, Hold
- **Path:** Tile angles 0–359, midspin 999, Twirl (reverse), MultiPlanet
- **Camera/Track:** MoveCamera, PositionTrack, MoveTrack, ColorTrack, AnimateTrack
- **Gameplay:** Checkpoint, AutoPlayTiles, SetPlanetRotation, FreeRoam*, ScaleMargin, ScaleRadius, Multitap, Hide, KillPlayer
- **Audio:** SetHitsound, PlaySound, SetHoldSound
- **Control flow:** RepeatEvents, SetConditionalEvents, SetInputEvent
- **VFX:** Flash, Bloom, ShakeScreen, SetFilter
- **Conditioning:** Difficulty prefix token (from `settings.difficulty` or index JSON)

**NOT included:** Decorations, particles, editor-only events. No style/descriptor tags.

### Hardware Requirements

**Smoke test (Whisper-tiny):**
- **GPU:** Any (4060 8GB, T4, etc.) or CPU
- **VRAM:** ~4 GB
- **Time:** ~5-10 minutes for 100 steps
- **Purpose:** Verify pipeline, vocab emission (speed/twirl/camera/VFX/diff tokens)

**Full training (Whisper-small production):**
- **GPU:** Colab Pro A100/L4 (24GB VRAM) OR 4090 (24GB) OR 4× GPUs with 8GB+ each
- **VRAM:** ~20-24 GB per GPU (single GPU: microbatch 8-16; multi-GPU: batch 32 per GPU)
- **Time:** ~80-200 GPU-hours (depends on dataset size and convergence)
- **NOT feasible:** Single 4060 8GB (insufficient VRAM for Whisper-small full batch)

### Exact Training Commands

**Smoke test (Whisper-tiny, CPU or any GPU):**

```bash
# Extract dataset
tar -xzf adofai-top100.tar.gz -C datasets/

# Run smoke (100 steps, tiny model, 5 charts × 8 rotations = 40 variants)
python osuT5/train.py -cn adofai_whisper_tiny \
  data.train_dataset_path=datasets/adofai-top100 \
  data.test_dataset_path=datasets/adofai-top100
```

**Full production training (Whisper-small, A100/L4/4090):**

```bash
# Single GPU (microbatch 8-16):
python osuT5/train.py -cn adofai_v31 \
  data.train_dataset_path=datasets/adofai-top100 \
  data.test_dataset_path=datasets/adofai-top100 \
  optim.batch_size=8

# Multi-GPU (4 GPUs, batch 32 per GPU = 128 total):
python osuT5/train.py -cn adofai_v31 \
  data.train_dataset_path=datasets/adofai-top100 \
  data.test_dataset_path=datasets/adofai-top100
```

**Output:**
- Checkpoints: `outputs/<timestamp>/checkpoints/`
- Logs: `tensorboard_logs/`
- Final vocab size: ~10k-15k tokens (depends on event range quantization)

### Generation (two-pass, like osu!)

After training, generate charts using osuT5 inference:

```bash
python osuT5/inference.py \
  checkpoint_path=outputs/<timestamp>/checkpoints/step_65536.pt \
  audio_path=your_song.ogg \
  output_path=./generated \
  format=adofai \
  difficulty=12
```

Generation is **staged** (v29/v31 style):
1. **Pass 1:** Timing context (BPM, offset, SetSpeed, timing points)
2. **Pass 2:** Full chart (tiles, angles, midspin, Twirl, camera, VFX) conditioned on Pass 1

**Note:** Generation code for ADOFAI is in progress. Current path: osuT5 model outputs tokens → converter decodes to Events → `adofai/converter.py` events_to_level() → write .adofai file.

## Evaluation Ideas

Metrics to assess generated ADOFAI charts:
- **Timing accuracy**: Do tiles align with beats/onsets?
- **Playability**: Can a human play the chart?
- **Angle variety**: Distribution of angles (avoid repetitive patterns)
- **Speed coherence**: Do SetSpeed events match song intensity?
- **Style consistency**: Does output match conditioning (BPM, difficulty)?

Manual playtesting is essential — ADOFAI is very sensitive to timing precision.

## Current Limitations (v1)

- ❌ Model NOT trained on ADOFAI data (export/inference is stub/placeholder only)
- ❌ LSTM proof-of-concept (Whisper encoder integration TODO)
- ❌ No camera events (MoveCamera, MoveTrack)
- ❌ No VFX events (Flash, Bloom, ShakeScreen)
- ❌ No decorations
- ❌ No multi-file level support (separate audio/image files)
- ✅ Training pipeline complete (dataset, tokenizer, model, checkpoints)
- ✅ Memory-optimized for T4 GPU (log-mel spectrograms, 60s audio cap)
- ✅ Basic playable mechanics ready for training (tiles, speed, twirl, hold)

## Next Steps

1. **Collect ADOFAI dataset** — gather 100+ Workshop charts + audio ✅ (you can start now)
2. **Run training** — use Colab notebook or local GPU ✅ (pipeline ready)
3. **Integrate Whisper encoder** — adapt from upstream osuT5 (future work)
4. **Evaluate** — playtest generated charts and iterate
5. **Extend to v2** — add camera, VFX, decorations

## References

- **ADOFAI Workshop**: Steam Workshop for community charts
- **ADOFAI-JS**: https://github.com/adofaiex/ADOFAI-JS — reference parser
- **adofaipy**: https://github.com/M1n3c4rt/adofaipy — Python parser
- **Format documentation**: https://fileformat.fandom.com/wiki/Adofai

## Example: Minimal Valid ADOFAI

See `fixtures/sample.adofai` for a complete example.

```json
{
    "settings": {
        "version": 14,
        "bpm": 120,
        "offset": 0,
        "songFilename": "song.ogg",
        ...
    },
    "angleData": [0, 0, 90, 180],
    "actions": [],
    "decorations": []
}
```

This creates a simple 4-tile path: right, right, up, left.
