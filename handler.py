#!/usr/bin/env python3
"""
RunPod Handler for Mapperatorinator V30
Serverless endpoint for generating osu! beatmaps from audio files
"""

import base64
import io
import os
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

from config import InferenceConfig
from inference import (
    prepare_args, load_model, load_diff_model, get_args_from_beatmap,
    get_config, generate, autofill_paths
)
from osuT5.osuT5.tokenizer import ContextType


def init_models():
    """Initialize models on startup"""
    global model, tokenizer, diff_model, diff_tokenizer, refine_model, args
    
    print("Initializing Mapperatorinator V30 models...")
    
    # Initialize Hydra configuration
    with initialize(config_path="configs/inference", version_base="1.1"):
        args = compose(config_name="v30")
    
    # Convert to InferenceConfig object for better type handling
    args = OmegaConf.structured(InferenceConfig(**args))
    
    # Force specific settings for serverless endpoint
    args.export_osz = True  # Always export .osz files
    args.use_server = False  # Don't use server mode in serverless
    args.add_to_beatmap = False  # Always create new beatmap
    args.output_type = [ContextType.MAP]  # Always generate maps
    args.in_context = []  # No context by default
    
    prepare_args(args)
    print(f"Using device: {args.device}")
    
    # Load main model
    print("Loading main Mapperatorinator model...")
    model, tokenizer = load_model(
        args.model_path, 
        args.train, 
        args.device, 
        args.max_batch_size, 
        args.use_server
    )
    print("✓ Main model loaded successfully")
    
    # Load diffusion models if needed
    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        print("Loading diffusion models...")
        try:
            diff_model, diff_tokenizer = load_diff_model(
                args.diff_ckpt, 
                args.diffusion, 
                args.device
            )
            print("✓ Diffusion model loaded successfully")
            
            if os.path.exists(args.diff_refine_ckpt):
                refine_model = load_diff_model(
                    args.diff_refine_ckpt, 
                    args.diffusion, 
                    args.device
                )[0]
                print("✓ Refine model loaded successfully")
            
            if args.compile:
                print("Compiling diffusion model...")
                diff_model.forward = torch.compile(
                    diff_model.forward, 
                    mode="reduce-overhead", 
                    fullgraph=True
                )
                print("✓ Model compilation complete")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load diffusion models: {e}")
            print("   Position generation will be disabled")
            args.generate_positions = False
    
    print("✅ All models initialized successfully")
    
    # Print memory usage if available
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")


def validate_audio_format(audio_data: bytes) -> bool:
    """Validate that the audio data is in a supported format"""
    # Check for common audio file headers
    audio_headers = [
        b'\xFF\xFB',  # MP3
        b'\xFF\xF3',  # MP3
        b'\xFF\xF2',  # MP3
        b'RIFF',      # WAV
        b'OggS',      # OGG
        b'fLaC',      # FLAC
        b'\x00\x00\x00\x20ftypM4A',  # M4A
    ]
    
    for header in audio_headers:
        if audio_data.startswith(header):
            return True
    
    # Additional check for WAV files
    if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
        return True
        
    return False


def validate_beatmap_format(beatmap_data: str) -> bool:
    """Validate that the beatmap data is a valid .osu file"""
    lines = beatmap_data.strip().split('\n')
    if not lines:
        return False
    
    # Check for osu file header
    first_line = lines[0].strip()
    return first_line.startswith('osu file format v')


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Expected input format:
    {
        "input": {
            "audio": "base64_encoded_audio_file",
            "beatmap": "base64_encoded_osu_file" (optional),
            "title": "Song Title" (optional),
            "artist": "Artist Name" (optional),
            "creator": "Mapper Name" (optional, defaults to 'Mapperatorinator'),
            "version": "Difficulty Name" (optional, defaults to 'Mapperatorinator V30'),
            "gamemode": 0 (optional, 0=osu!, 1=taiko, 2=catch, 3=mania),
            "hp_drain_rate": 5.0 (optional, HP drain rate),
            "circle_size": 4.0 (optional, circle size),
            "overall_difficulty": 8.0 (optional, overall difficulty),
            "approach_rate": 9.0 (optional, approach rate),
            "slider_multiplier": 1.4 (optional, slider velocity multiplier),
            "slider_tick_rate": 1.0 (optional, slider tick rate),
            "keycount": 4 (optional, for mania mode),
            "hitsounded": true (optional, whether to include hitsounds),
            "temperature": 0.9 (optional, sampling temperature),
            "cfg_scale": 1.0 (optional, classifier-free guidance scale),
            "top_p": 0.9 (optional, top-p sampling),
            "descriptors": ["stream", "jump"] (optional, style descriptors),
            "negative_descriptors": ["easy"] (optional, descriptors to avoid),
            "seed": 12345 (optional, random seed),
            "generate_positions": false (optional, use diffusion for positions)
        }
    }
    
    Returns:
    {
        "osz_file": "base64_encoded_osz_file",
        "filename": "generated_beatmap.osz",
        "metadata": {
            "title": "Song Title",
            "artist": "Artist Name",
            "creator": "Mapper Name",
            "version": "Difficulty Name"
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        # Validate required inputs
        if "audio" not in input_data:
            return {"error": "Missing required 'audio' field in input"}
        
        # Decode audio data
        try:
            audio_data = base64.b64decode(input_data["audio"])
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {str(e)}"}
        
        # Validate audio format
        if not validate_audio_format(audio_data):
            return {"error": "Unsupported audio format. Please use MP3, WAV, OGG, FLAC, or M4A"}
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Save audio file
            audio_path = temp_dir_path / "audio.mp3"  # Default to mp3 extension
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            
            # Handle optional beatmap reference
            beatmap_path = None
            if "beatmap" in input_data:
                try:
                    beatmap_data = base64.b64decode(input_data["beatmap"]).decode('utf-8')
                    if validate_beatmap_format(beatmap_data):
                        beatmap_path = temp_dir_path / "reference.osu"
                        with open(beatmap_path, "w", encoding='utf-8') as f:
                            f.write(beatmap_data)
                    else:
                        return {"error": "Invalid beatmap format. Must be a valid .osu file"}
                except Exception as e:
                    return {"error": f"Invalid base64 beatmap data: {str(e)}"}
            
            # Create inference config with user inputs
            inference_args = InferenceConfig()
            
            # Copy base config from v30
            for key, value in args.items():
                if hasattr(inference_args, key):
                    setattr(inference_args, key, value)
            
            # Override with user inputs
            inference_args.audio_path = str(audio_path)
            inference_args.output_path = str(temp_dir_path)
            inference_args.beatmap_path = str(beatmap_path) if beatmap_path else ""
            inference_args.export_osz = True
            
            # Set metadata
            inference_args.title = input_data.get("title", "")
            inference_args.artist = input_data.get("artist", "")
            inference_args.creator = input_data.get("creator", "Mapperatorinator")
            inference_args.version = input_data.get("version", "Mapperatorinator V30")
            
            # Set optional parameters
            if "gamemode" in input_data:
                inference_args.gamemode = input_data["gamemode"]
            if "hp_drain_rate" in input_data:
                inference_args.hp_drain_rate = input_data["hp_drain_rate"]
            if "circle_size" in input_data:
                inference_args.circle_size = input_data["circle_size"]
            if "overall_difficulty" in input_data:
                inference_args.overall_difficulty = input_data["overall_difficulty"]
            if "approach_rate" in input_data:
                inference_args.approach_rate = input_data["approach_rate"]
            if "slider_multiplier" in input_data:
                inference_args.slider_multiplier = input_data["slider_multiplier"]
            if "slider_tick_rate" in input_data:
                inference_args.slider_tick_rate = input_data["slider_tick_rate"]
            if "keycount" in input_data:
                inference_args.keycount = input_data["keycount"]
            if "hitsounded" in input_data:
                inference_args.hitsounded = input_data["hitsounded"]
            if "temperature" in input_data:
                inference_args.temperature = input_data["temperature"]
            if "cfg_scale" in input_data:
                inference_args.cfg_scale = input_data["cfg_scale"]
            if "top_p" in input_data:
                inference_args.top_p = input_data["top_p"]
            if "descriptors" in input_data:
                inference_args.descriptors = input_data["descriptors"]
            if "negative_descriptors" in input_data:
                inference_args.negative_descriptors = input_data["negative_descriptors"]
            if "seed" in input_data:
                inference_args.seed = input_data["seed"]
            if "generate_positions" in input_data:
                inference_args.generate_positions = input_data["generate_positions"]
            
            # Validate and autofill paths
            result = autofill_paths(inference_args)
            if not result['success']:
                return {"error": f"Path validation failed: {'; '.join(result['errors'])}"}
            
            # Get metadata from beatmap if provided
            get_args_from_beatmap(inference_args, tokenizer)
            
            # Generate config
            generation_config, beatmap_config = get_config(inference_args)
            
            # Generate beatmap
            beatmap_result, result_path, osz_path = generate(
                inference_args,
                generation_config=generation_config,
                beatmap_config=beatmap_config,
                model=model,
                tokenizer=tokenizer,
                diff_model=diff_model,
                diff_tokenizer=diff_tokenizer,
                refine_model=refine_model,
                verbose=True
            )
            
            if not osz_path or not Path(osz_path).exists():
                return {"error": "Failed to generate .osz file"}
            
            # Read and encode the .osz file
            with open(osz_path, "rb") as f:
                osz_data = f.read()
            
            osz_base64 = base64.b64encode(osz_data).decode('utf-8')
            
            # Create filename
            title = inference_args.title or "Generated"
            artist = inference_args.artist or "Unknown"
            filename = f"{artist} - {title}.osz"
            
            # Clean filename for filesystem compatibility
            import re
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            return {
                "osz_file": osz_base64,
                "filename": filename,
                "metadata": {
                    "title": inference_args.title,
                    "artist": inference_args.artist,
                    "creator": inference_args.creator,
                    "version": inference_args.version,
                    "gamemode": inference_args.gamemode,
                    "seed": inference_args.seed
                }
            }
            
    except Exception as e:
        # Log the full traceback for debugging
        error_msg = f"Handler error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}


def runpod_handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler that ensures models are initialized
    """
    # Check if models are initialized
    if not hasattr(runpod_handler, '_models_initialized'):
        print("Initializing models...")
        init_models()
        runpod_handler._models_initialized = True
        print("Models initialization complete")
    
    return handler(event)


if __name__ == "__main__":
    # For local testing
    print("Initializing models for local testing...")
    init_models()
    print("Models initialized. Ready for inference.")
    
    # Example test
    test_event = {
        "input": {
            "audio": "",  # Add base64 encoded audio here for testing
            "title": "Test Song",
            "artist": "Test Artist",
            "temperature": 0.9,
            "cfg_scale": 1.0
        }
    }
    
    # Uncomment to test locally
    # result = handler(test_event)
    # print(result)
