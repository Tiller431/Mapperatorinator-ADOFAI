# RunPod Deployment Guide for Mapperatorinator V30

## Quick Deployment Steps

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -f Dockerfile.runpod -t your-username/mapperatorinator-v30-runpod .

# Push to Docker Hub (or your preferred registry)
docker push your-username/mapperatorinator-v30-runpod
```

### 2. Deploy on RunPod

1. Go to [RunPod](https://runpod.io) and create an account
2. Navigate to Serverless → Endpoints
3. Click "New Endpoint"
4. Fill in the details:
   - **Name**: Mapperatorinator V30
   - **Docker Image**: `your-username/mapperatorinator-v30-runpod`
   - **Container Disk**: 20GB (minimum for models)
   - **Memory**: 16GB (minimum for V30 model)
   - **GPU**: Any GPU with 8GB+ VRAM (RTX 3080, A100, etc.)

### 3. Environment Variables (Optional)

You can set these in the RunPod environment if needed:
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility
- `PYTORCH_CUDA_ALLOC_CONF`: GPU memory management
- `HYDRA_FULL_ERROR`: Set to 1 for detailed error messages

### 4. Test the Endpoint

Use the test script or make HTTP requests:

```python
import requests
import base64

# Your RunPod endpoint URL
endpoint_url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"

# Encode audio file
with open("test_audio.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Request payload
payload = {
    "input": {
        "audio": audio_data,
        "title": "My Song",
        "artist": "My Artist",
        "temperature": 0.9,
        "cfg_scale": 1.0
    }
}

# Headers with your API key
headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
    "Content-Type": "application/json"
}

# Make request
response = requests.post(endpoint_url, json=payload, headers=headers)
result = response.json()

# Save result
if "output" in result and "osz_file" in result["output"]:
    osz_data = base64.b64decode(result["output"]["osz_file"])
    with open(result["output"]["filename"], "wb") as f:
        f.write(osz_data)
    print(f"Generated: {result['output']['filename']}")
```

## Performance Notes

### Cold Start Optimization
- Models are loaded on first request to minimize cold start time
- Subsequent requests will be much faster
- Consider using RunPod's "Always On" workers for production

### Memory Requirements
- **Minimum**: 16GB RAM, 8GB GPU VRAM
- **Recommended**: 32GB RAM, 16GB GPU VRAM
- **With Diffusion**: Additional 4GB GPU VRAM

### Processing Time
- **Audio Only**: ~30-60 seconds for a 3-minute song
- **With Diffusion**: ~60-120 seconds for a 3-minute song
- **Super Timing**: Additional 30-60 seconds

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase container memory or use smaller models
2. **Model Download Failures**: Check internet connectivity and model paths
3. **Audio Format Issues**: Ensure audio is in supported format (MP3, WAV, OGG, FLAC, M4A)
4. **Timeout**: Increase RunPod timeout settings for longer songs

### Debug Mode

Set environment variable `HYDRA_FULL_ERROR=1` for detailed error messages.

### Logs

Check RunPod logs for detailed error information:
- Model loading status
- Audio processing errors  
- Generation failures

## Cost Optimization

1. **Use Spot Instances**: Much cheaper for non-critical workloads
2. **Auto-scaling**: Set min workers to 0 to avoid idle costs
3. **Optimize Timeout**: Set appropriate timeout to avoid hanging requests
4. **Model Caching**: Pre-download models in Docker image to reduce cold start

## Security Notes

- Input validation is performed on audio and beatmap formats
- No file system access outside temp directories
- Base64 encoding ensures safe data transfer
- Consider rate limiting for production use

## API Limits

Current implementation supports:
- Audio files up to ~50MB (base64 encoded)
- Beatmap files up to ~1MB 
- No concurrent request limit (depends on RunPod plan)

For larger files, consider implementing file upload URLs instead of base64 encoding.
