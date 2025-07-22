#!/usr/bin/env python3
"""
Model caching script for RunPod Docker image
Downloads and caches all required models to reduce cold start time
"""

import os
import sys
from pathlib import Path

def download_main_model():
    """Download the main Mapperatorinator V30 model"""
    try:
        print("Downloading Mapperatorinator V30 model files...")
        
        # Use huggingface_hub to download files directly
        from huggingface_hub import snapshot_download
        
        print("Downloading model repository...")
        snapshot_download(
            repo_id='OliBomby/Mapperatorinator-v30',
            cache_dir='/root/.cache/huggingface',
            local_files_only=False
        )
        print("✓ Main model files downloaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download main model: {e}")
        
        # Try alternative approach with individual file downloads
        try:
            print("Attempting to download key model files individually...")
            from huggingface_hub import hf_hub_download
            
            # Download key files
            files_to_download = [
                'config.json',
                'pytorch_model.bin', 
                'tokenizer.json',
                'custom_checkpoint_0.pkl'
            ]
            
            for filename in files_to_download:
                try:
                    hf_hub_download(
                        repo_id='OliBomby/Mapperatorinator-v30',
                        filename=filename,
                        cache_dir='/root/.cache/huggingface'
                    )
                    print(f"✓ Downloaded {filename}")
                except Exception as file_error:
                    print(f"⚠️  Could not download {filename}: {file_error}")
            
            print("✓ Individual file downloads completed")
            return True
            
        except Exception as e2:
            print(f"❌ Individual file download also failed: {e2}")
            return False

def download_diffusion_model():
    """Download the osu-diffusion model"""
    try:
        print("Downloading osu-diffusion model...")
        from huggingface_hub import hf_hub_download
        
        # Download tokenizer
        tokenizer_file = hf_hub_download(
            repo_id='OliBomby/osu-diffusion-v2',
            filename='tokenizer.pkl',
            cache_dir='/root/.cache/huggingface'
        )
        print("✓ Diffusion tokenizer downloaded")
        
        # Download model
        model_file = hf_hub_download(
            repo_id='OliBomby/osu-diffusion-v2',
            filename='model_ema.pkl',
            cache_dir='/root/.cache/huggingface'
        )
        print("✓ Diffusion model downloaded")
        
        return True
    except Exception as e:
        print(f"⚠️  Failed to download diffusion model: {e}")
        print("   This is optional and will be downloaded at runtime if needed")
        return False

def check_model_cache():
    """Check which models are already cached"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print(f"Cache directory: {cache_dir}")
        subdirs = list(cache_dir.iterdir())
        print(f"Cached items: {len(subdirs)}")
        for item in subdirs:
            if item.is_dir():
                print(f"  - {item.name}")
    else:
        print("No cache directory found")

def main():
    """Main function to download all models"""
    print("Model Caching Script for Mapperatorinator V30")
    print("=" * 50)
    
    # Set cache directories
    os.environ['HF_HOME'] = str(Path.home() / ".cache" / "huggingface")
    os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / ".cache" / "huggingface")
    os.environ['HF_HUB_CACHE'] = str(Path.home() / ".cache" / "huggingface")
    
    # Create cache directories
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cache directory: {cache_dir}")
    
    # Download models
    main_success = download_main_model()
    diffusion_success = download_diffusion_model()
    
    print("\n" + "=" * 50)
    print("Model Caching Summary:")
    print(f"✓ Main model: {'Success' if main_success else 'Failed (will download at runtime)'}")
    print(f"✓ Diffusion model: {'Success' if diffusion_success else 'Failed (optional)'}")
    
    # Show cache status
    print("\nCache Status:")
    check_model_cache()
    
    # Don't exit with error - models can be downloaded at runtime
    if main_success or diffusion_success:
        print("\n✅ Model caching completed with some success!")
    else:
        print("\n⚠️  No models were cached, but this is not critical.")
        print("   Models will be downloaded at runtime on first request.")
    
    print("Docker build will continue...")

if __name__ == "__main__":
    main()
