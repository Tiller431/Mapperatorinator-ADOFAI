# RunPod Serverless Handler for Mapperatorinator V30

This handler provides a serverless endpoint for generating osu! beatmaps using the Mapperatorinator V30 model.

## Features

- **Fixed Configuration**: Always uses the V30 model configuration
- **Audio Input**: Accepts base64-encoded audio files (MP3, WAV, OGG, FLAC, M4A)
- **Optional Reference**: Can use an existing .osu file as reference for metadata
- **Customizable Parameters**: Supports all major beatmap parameters
- **OSZ Output**: Always outputs complete .osz files ready for osu!

## Input Format

```json
{
    "input": {
        "audio": "base64_encoded_audio_file",
        "beatmap": "base64_encoded_osu_file",  // optional
        "title": "Song Title",               // optional
        "artist": "Artist Name",             // optional
        "creator": "Mapper Name",            // optional, defaults to 'Mapperatorinator'
        "version": "Difficulty Name",        // optional, defaults to 'Mapperatorinator V30'
        "gamemode": 0,                       // optional, 0=osu!, 1=taiko, 2=catch, 3=mania
        "hp_drain_rate": 5.0,               // optional, HP drain rate
        "circle_size": 4.0,                 // optional, circle size
        "overall_difficulty": 8.0,          // optional, overall difficulty
        "approach_rate": 9.0,               // optional, approach rate
        "slider_multiplier": 1.4,           // optional, slider velocity multiplier
        "slider_tick_rate": 1.0,            // optional, slider tick rate
        "keycount": 4,                      // optional, for mania mode
        "hitsounded": true,                 // optional, whether to include hitsounds
        "temperature": 0.9,                 // optional, sampling temperature
        "cfg_scale": 1.0,                   // optional, classifier-free guidance scale
        "top_p": 0.9,                       // optional, top-p sampling
        "descriptors": ["stream", "jump"],   // optional, style descriptors
        "negative_descriptors": ["easy"],    // optional, descriptors to avoid
        "seed": 12345,                      // optional, random seed
        "generate_positions": false         // optional, use diffusion for positions
    }
}
```

## Output Format

```json
{
    "osz_file": "base64_encoded_osz_file",
    "filename": "Artist - Title.osz",
    "metadata": {
        "title": "Song Title",
        "artist": "Artist Name",
        "creator": "Mapper Name",
        "version": "Difficulty Name",
        "gamemode": 0,
        "seed": 12345
    }
}
```

## Error Handling

If an error occurs, the response will be:

```json
{
    "error": "Error description with details"
}
```

## Model Configuration

The handler uses the V30 configuration with these fixed settings:
- Always exports .osz files
- Always generates new beatmaps (doesn't modify existing ones)
- Uses MAP output type
- No context by default

## Deployment Notes

1. Ensure all dependencies from `requirements_runpod.txt` are installed
2. The main model files should be accessible at the paths specified in the V30 config
3. GPU memory requirements depend on the model size and diffusion settings
4. The handler initializes models on first request to optimize cold start performance

## Usage Example

```python
import base64
import requests

# Encode your audio file
with open("song.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Prepare the request
payload = {
    "input": {
        "audio": audio_data,
        "title": "My Song",
        "artist": "My Artist",
        "temperature": 0.9,
        "cfg_scale": 1.0,
        "gamemode": 0
    }
}

# Send request to RunPod endpoint
response = requests.post("YOUR_RUNPOD_ENDPOINT", json=payload)
result = response.json()

# Save the generated .osz file
if "osz_file" in result:
    osz_data = base64.b64decode(result["osz_file"])
    with open(result["filename"], "wb") as f:
        f.write(osz_data)
```
