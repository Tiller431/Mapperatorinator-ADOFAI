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

### Dataset Needs

1. **ADOFAI charts + audio pairs**:
   - Collect `.adofai` files with corresponding audio (`.ogg`, `.mp3`, etc.)
   - Recommended: 500+ charts covering variety of styles, BPMs, difficulties
   - Sources: ADOFAI Workshop, community chart collections, custom levels

2. **Metadata**:
   - BPM, offset, difficulty rating (if available)
   - Chart style descriptors (fast, slow, twirl-heavy, etc.) — optional but helpful

3. **Quality filtering**:
   - Ensure charts are playable and well-timed
   - Remove broken/corrupted files
   - Consider difficulty distribution (easy, medium, hard)

### Data Preparation

Create a dataset using a script similar to osu!'s `Mapperator.ConsoleApp`:
- Parse `.adofai` files → extract events
- Generate spectrograms from audio
- Pair events with audio windows
- Create train/val/test splits

### Model Retraining

Adapt the osuT5 training pipeline:
1. Replace osu! parser with ADOFAI parser
2. Update tokenizer vocabulary for ADOFAI events
3. Adjust conditioning tokens (BPM, difficulty, style)
4. Train encoder-decoder on audio → ADOFAI events

Expected compute: Similar to osu! training (~2500 GPU hours for full quality, but smaller datasets may suffice for initial results).

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
