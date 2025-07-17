#!/bin/bash
# Start vLLM server for SmolDocling Document Understanding

echo "Starting vLLM server for SmolDocling..."

# Kill any existing process on port 8002
lsof -ti:8002 | xargs kill -9 2>/dev/null || true

# Activate virtual environment
source .venv/bin/activate

# Start vLLM with SmolDocling model
# SmolDocling ist ein spezielles Document Understanding Modell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "microsoft/Florence-2-base" \
    --port 8002 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto \
    --served-model-name "smoldocling" &

# Wait for server to start
echo "Waiting for vLLM server to start..."
sleep 10

# Test if server is running
curl -s http://localhost:8002/v1/models > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ vLLM SmolDocling server started successfully on port 8002"
else
    echo "❌ Failed to start vLLM server"
fi