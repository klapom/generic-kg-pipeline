#!/usr/bin/env python3
"""
Test SmolDocling with a simple text image
"""

import logging
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_simple_test_image():
    """Create a simple test image with clear text"""
    # Create white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Add simple text
    text_lines = [
        "Test Document",
        "",
        "This is a simple test document.",
        "It contains clear text that should be easy to read.",
        "",
        "Features:",
        "- Clear font",
        "- High contrast", 
        "- Simple layout",
        "",
        "End of document."
    ]
    
    # Draw text
    y_position = 50
    for line in text_lines:
        draw.text((50, y_position), line, fill='black')
        y_position += 30
    
    # Save image
    image.save('test_simple.png')
    logger.info("Created test_simple.png")
    return image

def test_with_simple_image():
    """Test SmolDocling with simple image"""
    try:
        # Create test image
        logger.info("Creating simple test image...")
        test_image = create_simple_test_image()
        
        # Import and create SmolDocling client
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        
        logger.info("📦 Creating SmolDocling client...")
        client = VLLMSmolDoclingClient()
        
        # Ensure model is loaded
        if not client.model_manager.is_model_loaded(client.model_id):
            logger.info("Loading model...")
            client.model_manager.load_model(client.model_id)
        
        logger.info("🔄 Processing simple test image...")
        page = client.process_pdf_page(test_image, 1)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 SIMPLE IMAGE TEST RESULT:")
        logger.info(f"{'='*80}")
        logger.info(f"📝 Extracted text ({len(page.text)} chars):")
        logger.info(f"--- START ---")
        logger.info(page.text)
        logger.info(f"--- END ---")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting simple image test...")
    success = test_with_simple_image()
    if success:
        logger.info("🎉 Test completed!")
    else:
        logger.error("💥 Test failed!")