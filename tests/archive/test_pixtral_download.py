#!/usr/bin/env python3
"""
Test Pixtral Download and Integration

Downloads and tests Pixtral-12B model.
"""

import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
log_dir = Path("tests/debugging/pixtral_test")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'test_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, '.')

def test_pixtral_availability():
    """Test if Pixtral model can be downloaded"""
    from transformers import AutoProcessor
    
    logger.info("Testing Pixtral model availability...")
    
    try:
        # Try to load the processor first (smaller download)
        processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        logger.info("✅ Pixtral processor loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Pixtral processor: {e}")
        return False

def test_pixtral_client():
    """Test Pixtral client"""
    from core.clients.transformers_pixtral_client import TransformersPixtralClient
    from core.parsers.interfaces.data_models import VisualElementType
    
    logger.info("Testing Pixtral client...")
    
    # Initialize client with 8-bit quantization for memory efficiency
    client = TransformersPixtralClient(
        temperature=0.2,
        max_new_tokens=512,
        load_in_8bit=True
    )
    
    # Load test image
    test_image_path = Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return False
        
    logger.info(f"Loading test image: {test_image_path}")
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    # Run analysis
    logger.info("Starting Pixtral analysis...")
    try:
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        logger.info("\n" + "="*60)
        logger.info("PIXTRAL ANALYSIS RESULTS")
        logger.info("="*60)
        logger.info(f"Success: {result.success}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Processing Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"Description: {result.description}")
        if result.ocr_text:
            logger.info(f"OCR Text: {result.ocr_text}")
        if result.error_message:
            logger.error(f"Error: {result.error_message}")
        
        # Save results
        results_file = log_dir / "pixtral_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "success": result.success,
                "confidence": result.confidence,
                "processing_time": result.processing_time_seconds,
                "description": result.description,
                "ocr_text": result.ocr_text,
                "error_message": result.error_message,
                "metadata": result.metadata
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Cleanup
        client.cleanup()
        
        return result.success
        
    except Exception as e:
        logger.error(f"Pixtral test failed: {e}", exc_info=True)
        client.cleanup()
        return False

def main():
    """Run Pixtral tests"""
    
    logger.info("Starting Pixtral download and integration test...")
    
    # Step 1: Check availability
    if not test_pixtral_availability():
        logger.error("❌ Pixtral is not available")
        return
    
    # Step 2: Test client
    success = test_pixtral_client()
    
    if success:
        logger.info("✅ Pixtral test completed successfully!")
    else:
        logger.error("❌ Pixtral test failed!")

if __name__ == "__main__":
    main()