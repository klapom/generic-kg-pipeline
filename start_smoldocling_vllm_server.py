#!/usr/bin/env python3
"""
Start vLLM server for SmolDocling
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def start_vllm_server():
    """Start vLLM server for SmolDocling"""
    print("🚀 Starting vLLM server for SmolDocling...")
    
    # Kill any existing process on port 8002
    subprocess.run(["lsof", "-ti:8002"], capture_output=True)
    subprocess.run(["lsof", "-ti:8002", "|", "xargs", "kill", "-9"], shell=True, capture_output=True)
    
    # Start vLLM server
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "ds4sd/SmolDocling-256M-preview",
        "--port", "8002",
        "--gpu-memory-utilization", "0.2",
        "--max-model-len", "8192",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--served-model-name", "smoldocling"
    ]
    
    # Start server in background
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    max_retries = 60  # 60 seconds
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8002/v1/models")
            if response.status_code == 200:
                print("✅ vLLM SmolDocling server is ready!")
                print(f"Models available: {response.json()}")
                return process
        except:
            pass
        time.sleep(1)
        if i % 10 == 0:
            print(f"   Still waiting... ({i}s)")
    
    print("❌ Server failed to start")
    process.terminate()
    return None

if __name__ == "__main__":
    process = start_vllm_server()
    if process:
        print("\n📝 Server is running. Press Ctrl+C to stop.")
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()