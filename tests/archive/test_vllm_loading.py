#!/usr/bin/env python3
"""
Test vLLM Model Loading
"""

import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vllm_model_loading():
    """Test just loading the vLLM SmolDocling model"""
    try:
        logger.info("🧪 Testing vLLM SmolDocling model loading...")
        
        # Import and create client
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        
        logger.info("📦 Creating VLLMSmolDoclingClient...")
        client = VLLMSmolDoclingClient()
        
        logger.info("🔧 Ensuring model is loaded...")
        success = client.ensure_model_loaded()
        
        if success:
            logger.info("✅ Model loaded successfully!")
            return True
        else:
            logger.error("❌ Model failed to load!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting vLLM model loading test...")
    success = test_vllm_model_loading()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")