#!/usr/bin/env python3
"""
Test vLLM SmolDocling Multimodal Inference
"""

import logging
import os
from pathlib import Path
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_multimodal_inference():
    """Test multimodal inference with a simple image"""
    try:
        logger.info("🧪 Testing vLLM SmolDocling multimodal inference...")
        
        # Import and create client
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        
        logger.info("📦 Creating VLLMSmolDoclingClient...")
        client = VLLMSmolDoclingClient()
        
        logger.info("🔧 Ensuring model is loaded...")
        success = client.ensure_model_loaded()
        
        if not success:
            logger.error("❌ Model failed to load!")
            return False
            
        logger.info("✅ Model loaded successfully!")
        
        # Create a simple test image (solid color)
        logger.info("🖼️ Creating test image...")
        test_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # Test processing a single page
        logger.info("🔄 Testing page processing...")
        result = client.process_pdf_page(test_image, 1)
        
        logger.info(f"✅ Page processing completed!")
        logger.info(f"📊 Result: {len(result.text)} chars text, {len(result.tables)} tables")
        logger.info(f"📝 Text preview: {result.text[:100]}...")
        
        return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting vLLM multimodal inference test...")
    success = test_multimodal_inference()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")