#!/usr/bin/env python3
"""
Simplified model pre-download script for RunPod Docker image
Just downloads model files without trying to load them
"""

import os
import sys
from pathlib import Path

def download_model_files():
    """Download model files using huggingface_hub"""
    try:
        print("Setting up model cache directories...")
        
        # Set cache directories
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_HUB_CACHE'] = str(cache_dir)
        
        print(f"Cache directory: {cache_dir}")
        
        from huggingface_hub import snapshot_download, hf_hub_download
        
        # Download Mapperatorinator V30
        print("\n1. Downloading Mapperatorinator V30...")
        try:
            snapshot_download(
                repo_id='OliBomby/Mapperatorinator-v30',
                cache_dir=str(cache_dir),
                local_files_only=False
            )
            print("✅ Mapperatorinator V30 downloaded successfully")
            main_success = True
        except Exception as e:
            print(f"⚠️  Mapperatorinator V30 download failed: {e}")
            main_success = False
        
        # Download osu-diffusion v2
        print("\n2. Downloading osu-diffusion v2...")
        try:
            # Download key diffusion files
            hf_hub_download(
                repo_id='OliBomby/osu-diffusion-v2',
                filename='tokenizer.pkl',
                cache_dir=str(cache_dir)
            )
            hf_hub_download(
                repo_id='OliBomby/osu-diffusion-v2',
                filename='model_ema.pkl', 
                cache_dir=str(cache_dir)
            )
            print("✅ osu-diffusion v2 downloaded successfully")
            diffusion_success = True
        except Exception as e:
            print(f"⚠️  osu-diffusion v2 download failed: {e}")
            diffusion_success = False
        
        # Show final status
        print("\n" + "="*50)
        print("Model Download Summary:")
        print(f"• Mapperatorinator V30: {'✅ Downloaded' if main_success else '❌ Failed'}")
        print(f"• osu-diffusion v2: {'✅ Downloaded' if diffusion_success else '❌ Failed'}")
        
        # Show cache contents
        if cache_dir.exists():
            subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]
            print(f"\nCached repositories: {len(subdirs)}")
            for subdir in subdirs:
                if subdir.name.startswith('models--'):
                    repo_name = subdir.name.replace('models--', '').replace('--', '/')
                    print(f"  📦 {repo_name}")
        
        if main_success or diffusion_success:
            print("\n🎉 Model caching completed successfully!")
            print("Cold start times will be reduced.")
        else:
            print("\n⚠️  No models were cached, but this is not critical.")
            print("Models will be downloaded at runtime on first request.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fatal error in model download: {e}")
        print("Continuing with build anyway - models will download at runtime.")
        return False

if __name__ == "__main__":
    print("🚀 Model Pre-download Script for Mapperatorinator V30")
    print("="*60)
    
    success = download_model_files()
    
    print("\n✅ Build process continuing...")
    # Always exit successfully to not break Docker build
    sys.exit(0)
