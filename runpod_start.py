#!/usr/bin/env python3
"""
RunPod startup script for Mapperatorinator V30
"""

import runpod
from handler import runpod_handler

if __name__ == "__main__":
    print("Starting RunPod serverless endpoint for Mapperatorinator V30")
    runpod.serverless.start({"handler": runpod_handler})
