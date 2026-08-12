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

## Training Requirements

To generate quality ADOFAI charts, the model must be retrained. The current pretrained weights are for osu! and will not work well for ADOFAI.

### Dataset Layout

The training pipeline expects charts in Workshop-style folders:

```
<data_root>/
  <workshopId>__<chartName>/
    level.adofai
    <audio>.ogg|.mp3|.wav|.flac
  
  another_workshop_id__ChartName/
    level.adofai
    song.ogg
  
  ...
```

**Optional:** Create an index JSON with metadata:
```json
[
  {
    "workshop_id": "1234567890",
    "chart_dir": "1234567890__MyChart",
    "audio": "1234567890__MyChart/song.ogg",
    "has_audio": true
  },
  ...
]
```

### Dataset Needs

1. **ADOFAI charts + audio pairs**:
   - Collect `.adofai` files with corresponding audio (`.ogg`, `.mp3`, etc.)
   - Recommended: 100+ charts for initial training, 500+ for quality
   - Cover variety of styles, BPMs, difficulties
   - Sources: ADOFAI Workshop, community chart collections, custom levels

2. **Metadata** (extracted from `level.adofai`):
   - BPM, offset (from settings)
   - Tile angles, actions (from angleData/actions)
   - No additional metadata files required

3. **Quality filtering**:
   - Charts must have audio present
   - Remove broken/corrupted files
   - Ensure charts parse correctly with `adofai.parser`

### Training Options

**Option 1: Google Colab (Recommended for beginners)**

Use the provided Colab notebook for easy cloud training:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tiller431/Mapperatorinator-ADOFAI/blob/cursor/adofai-foundation-5317/colab/adofai_train_v1.ipynb)

**Features**:
- Free GPU (T4) or upgrade to Colab Pro
- No local setup required
- Stores data and checkpoints on Google Drive
- Step-by-step guided cells
- Automatic checkpoint saving

**Requirements**:
- Google account
- ADOFAI charts organized in Drive: `MyDrive/adofai-dataset/charts-top100/<id>__<name>/level.adofai` + audio

---

**Option 2: Local Training**

Test the training pipeline with minimal data:

```bash
# Create a small test dataset (3-5 charts)
mkdir -p test_data/test1__SimpleChart
cp your_chart.adofai test_data/test1__SimpleChart/level.adofai
cp your_audio.ogg test_data/test1__SimpleChart/audio.ogg

# Run smoke training (tiny model, few steps)
python3 -m adofai.train \
  --data_dir test_data \
  --output_dir adofai_smoke \
  --smoke \
  --device cpu
```

This will:
- Load 5 samples max
- Train tiny model (64 hidden dim, 1 layer)
- Run 2 epochs with batch size 2
- Save checkpoint to `adofai_smoke/`

### Running Full Training

For real training on a larger dataset:

```bash
python3 -m adofai.train \
  --data_dir /path/to/workshop_charts \
  --output_dir adofai_checkpoints \
  --batch_size 8 \
  --lr 1e-4 \
  --epochs 50 \
  --device cuda  # or cpu/mps
```

**Note:** The current training script uses a simple LSTM model for proof-of-concept. For production quality:
1. Integrate with osuT5 Whisper encoder architecture
2. Use spectrogram features instead of raw audio
3. Train on 100+ charts minimum
4. Expected compute: 100-500 GPU hours depending on dataset size

### Model Retraining

The v1 training script (`adofai/train.py`) provides:
- ✅ Dataset loading from Workshop folders
- ✅ Event tokenization (vocab size ~2000)
- ✅ Simple LSTM model for testing
- ❌ **TODO:** Full Whisper encoder integration (like osuT5)
- ❌ **TODO:** Spectrogram preprocessing
- ❌ **TODO:** Advanced training features (checkpointing, distributed, etc.)

For full-scale training, integrate with `osuT5/train.py` infrastructure.

## Evaluation Ideas

Metrics to assess generated ADOFAI charts:
- **Timing accuracy**: Do tiles align with beats/onsets?
- **Playability**: Can a human play the chart?
- **Angle variety**: Distribution of angles (avoid repetitive patterns)
- **Speed coherence**: Do SetSpeed events match song intensity?
- **Style consistency**: Does output match conditioning (BPM, difficulty)?

Manual playtesting is essential — ADOFAI is very sensitive to timing precision.

## Current Limitations (v1)

- ❌ Model not trained on ADOFAI data (stub generation only)
- ❌ No camera events (MoveCamera, MoveTrack)
- ❌ No VFX events (Flash, Bloom, ShakeScreen)
- ❌ No decorations
- ❌ No multi-file level support (separate audio/image files)
- ✅ Basic playable mechanics (tiles, speed, twirl, hold)

## Next Steps

1. **Collect ADOFAI dataset** — gather charts + audio
2. **Implement data loader** — adapt osuT5 dataset code for ADOFAI
3. **Retrain model** — fine-tune or train from scratch on ADOFAI data
4. **Evaluate** — playtest and iterate
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
