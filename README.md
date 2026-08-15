# Mapperatorinator-ADOFAI

**AI-powered chart generation for A Dance of Fire and Ice**

This is a fork of [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) by OliBomby, adapted to generate charts for [A Dance of Fire and Ice](https://store.steampowered.com/app/977950/A_Dance_of_Fire_and_Ice/) instead of osu! beatmaps.

## Current Status

Foundation (PR #1) and Whisper training (PR #2) are on `main`. Generate/export wiring uses the existing Hydra entry point — there is no `format=adofai` flag.

**Train:**
```bash
python osuT5/train.py -cn adofai_v31 data.train_dataset_path=/path/to/adofai-charts
```

**Generate** (writes a parseable `.adofai` via `events_to_level`; untrained weights make garbage charts):
```bash
python inference.py -cn adofai_v31 \
  audio_path=your_song.mp3 \
  output_path=./generated \
  model_path=/path/to/adofai_v31_checkpoint \
  difficulty=5
```

Empty / `scratch` `model_path` initializes Whisper-small plus a random ADOFAI head for a wiring smoke only.

**What works today:**
- Generate/export wiring: `python inference.py -cn adofai_v31` writes a parseable `.adofai` from an mp3 via Events → `events_to_level` (untrained weights = garbage charts)
- `.adofai` file I/O (UTF-8 BOM handling, trailing commas, `pathData`/`angleData`)
- Event vocabulary: tiles (angles 0-359°, midspin 999), SetSpeed, Twirl, Pause, Hold, MultiPlanet, camera (MoveCamera with `LastPositionNoRotation`), MoveTrack (`positionOffset`), VFX (Flash, Bloom, ShakeScreen, SetFilter)
- Training config: `configs/train/adofai_v31.yaml` (Whisper-small, difficulty conditioning ON, style/mapper/year/descriptors OFF, lossless rotate/reflect/pitch/rate augmentations, staged context: timing first, then map)
- Dataset: 126 Workshop charts archived at [Google Drive](https://drive.google.com/drive/folders/1lATJxQI8P3uLsRtiC7ay5u3SrFhH1cfd) (`adofai-top100.tar.gz`). Enough to smoke-test, not enough for quality generalization.

**What does not work:**
- No trained ADOFAI checkpoint exists yet. Inference will only generate meaningful charts after training on a large ADOFAI dataset.
- Decorations, editor-only actions, and tag/descriptor conditioning are deferred.
- `adofai/converter.py` timing calculations have known bugs; do not rely on its timestamp math as ground truth.

## Installation

The osu! Mapperatorinator install steps still apply (Python 3.10, ffmpeg, PyTorch, `requirements.txt`). ADOFAI train/generate commands are above.

### 1. Clone the repository

```sh
git clone https://github.com/Tiller431/Mapperatorinator-ADOFAI.git
cd Mapperatorinator-ADOFAI
```

### 2. (Optional) Create virtual environment

Use Python 3.10; later versions may not be compatible with dependencies.

```sh
python -m venv .venv

# Windows cmd.exe
.venv\Scripts\activate.bat
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux or macOS
source .venv/bin/activate
```

### 3. Install dependencies

- Python 3.10
- [Git](https://git-scm.com/downloads)
- [ffmpeg](http://www.ffmpeg.org/)
- [PyTorch](https://pytorch.org/get-started/locally/): Follow the Get Started guide to install `torch` and `torchaudio` with GPU support.

Then install remaining Python dependencies:

```sh
pip install -r requirements.txt
```

---

## Upstream osu! Mapperatorinator Reference

The sections below describe the **original osu! beatmap generator** on the `main` branch. They are kept for reference and will be replaced once ADOFAI work merges.

### Web GUI (osu! — Recommended for upstream)

For osu! beatmap generation, the Web UI provides a graphical interface.

```sh
python web-ui.py
```

- **Configure:** Set paths, gamemode, difficulty, style (year, mapper ID, descriptors), timing, hitsounds, etc.
- **Start/Cancel/Open Output:** Standard workflow controls.

The Web UI wraps `inference.py`.

### Command-Line Inference (osu!)

Run `inference.py` with [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). See `configs/inference/default.yaml` for parameters.

```sh
python inference.py \
  audio_path="path/to/audio.mp3" \
  output_path="output/" \
  gamemode=0 \
  difficulty=5.5 \
  year=2023 \
  descriptors="['jump aim','clean']" \
  in_context=[TIMING,KIAI]
```

**Example:**
```sh
python inference.py beatmap_path="'C:\Users\USER\AppData\Local\osu!\Songs\1 Kenji Ninuma - DISCO PRINCE\Kenji Ninuma - DISCOPRINCE (peppy) [Normal].osu'" gamemode=0 difficulty=5.5 year=2023 descriptors="['jump aim','clean']" in_context=[TIMING,KIAI]
```

### Interactive CLI (osu!)

```sh
chmod +x cli_inference.sh
./cli_inference.sh
```

Guided prompts for osu! beatmap parameters.

### Generation Tips (osu!)

- Edit `configs/inference/default.yaml` to set defaults.
- Descriptors: [osu! wiki beatmap tags](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags).
- Always provide `year` (2007–2023) and `difficulty` to avoid inconsistent generation.
- Increase `cfg_scale` to strengthen `mapper_id` and `descriptors` effects.
- Use `negative_descriptors` with `cfg_scale > 1` (must match descriptor count).
- Provide timing/kiai via `beatmap_path` and `in_context=[TIMING,KIAI]` for speed and accuracy.
- Remap part of a beatmap: `beatmap_path`, `start_time`, `end_time`, `add_to_beatmap=true`.
- Generate guest difficulty: `beatmap_path`, `in_context=[GD,TIMING,KIAI]`.
- Generate hitsounds only: `beatmap_path`, `in_context=[NO_HS,TIMING,KIAI]`.
- Generate timing only: `super_timing=true`, `output_type=[TIMING]`.

### MaiMod: AI-driven Modding Tool (osu!)

MaiMod detects issues in osu! beatmaps that automatic tools miss (incorrect snapping, timing, object placement, slider shapes, hitsounds).

Try [MaiMod Colab](https://colab.research.google.com/github/OliBomby/Mapperatorinator/blob/main/colab/mai_mod_inference.ipynb) or run locally:

```sh
python mai_mod.py beatmap_path="'C:\Users\USER\AppData\Local\osu!\Songs\1 Kenji Ninuma - DISCO PRINCE\Kenji Ninuma - DISCOPRINCE (peppy) [Normal].osu'"
```

Suggestions are ordered by "surprisal." Accepts same arguments as `inference.py`.

### Overview (osu! architecture)

**Tokenization:** osu! beatmaps → event representation (hit objects, hitsounds, slider velocities, timing, kiai). Quantized to 10ms intervals (time) and 32-pixel grids (position).

**Model architecture:** Wrapper around [HF Transformers Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) with custom input embeddings and loss (219M parameters). Mel spectrogram frames as encoder input; decoder outputs discrete event vocabulary.

**Multitask training format:** Conditional generation tokens (gamemode, difficulty, mapper ID, year, metadata) precede SOS. Random masking to 'unknown' tokens during training enables flexible inference metadata.

**Seamless long generation:** Context length: 8.192 seconds. 90% overlap, sequential generation, decoder pre-filled 50% from previous windows, logit processor prevents past/far-future tokens. Random offset training forces onset-based timing correction.

**Refined coordinates with diffusion:** Quantized positions (32px grid) denoised to final coordinates via modified [osu-diffusion](https://github.com/OliBomby/osu-diffusion). Specialized to last 10% of noise schedule. Slider end positions recalculated each diffusion step to match required slider lengths.

**Post-processing:** Refine with diffusion, resnap to ticks, snap overlaps, convert mania columns, generate taiko drumrolls, fix slider length discrepancies.

**Super timing generator:** Infer timing 20 times, average results. Near-perfect for variable BPM songs.

### Training (osu!)

Create dataset with [Mapperator console app](https://github.com/mappingtools/Mapperator/blob/master/README.md#create-a-high-quality-dataset). Requires [osu! OAuth token](https://osu.ppy.sh/home/account/edit).

```sh
Mapperator.ConsoleApp.exe dataset2 -t "/Mapperatorinator/datasets/beatmap_descriptors.csv" -i "path/to/osz/files" -o "/datasets/cool_dataset"
```

**Docker (recommended on WSL):**
```sh
docker compose up -d --force-recreate
docker attach mapperatorinator_space
```

**Train:**
```sh
python osuT5/train.py -cn train_v29 train_dataset_path="/workspace/datasets/cool_dataset" test_dataset_path="/workspace/datasets/cool_dataset" train_dataset_end=90 test_dataset_start=90 test_dataset_end=100
```

Configurations: `./configs/osut5/train.yaml`.

### See Also (osu!)

- [Mapper Classifier](./classifier/README.md)
- [RComplexion](./rcomplexion/README.md)

---

## Credits

**ADOFAI fork:** Tiller431

**Upstream Mapperatorinator:** [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) by OliBomby

Built upon:
- [osuT5](https://github.com/gyataro/osuT5) by gyataro — training code and T5 architecture
- [osu-diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby & NiceAesth — coordinate refinement

Special thanks (original Mapperatorinator):
1. The authors of [osuT5](https://github.com/gyataro/osuT5) for their training code.
2. Hugging Face team for their [tools](https://huggingface.co/docs/transformers/index).
3. [Jason Won](https://github.com/jaswon) and [Richard Nagyfi](https://github.com/sedthh) for bouncing ideas.
4. [Marvin](https://github.com/minetoblend) for donating training credits.
5. The osu! community for the beatmaps.

## Related Works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
3. [osu-diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)
4. [osuT5](https://github.com/gyataro/osuT5) by gyataro (Xiwen Teoh)
5. [Beat Learning](https://github.com/sedthh/BeatLearning) by sedthh (Richard Nagyfi)
6. [osu!dreamer](https://github.com/jaswon/osu-dreamer) by jaswon (Jason Won)
