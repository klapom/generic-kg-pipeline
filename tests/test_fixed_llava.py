#!/usr/bin/env python3
"""
Test Fixed LLaVA Client

Tests the improved LLaVA client with proper JSON parsing.
"""

import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
log_dir = Path("tests/debugging/fixed_llava_test")
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

from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.parsers.interfaces.data_models import VisualElementType

def main():
    """Test fixed LLaVA client"""
    
    logger.info("Testing fixed LLaVA-1.6-Mistral-7B client...")
    
    # Initialize client
    client = TransformersLLaVAClient(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_8bit=True,
        temperature=0.2,
        max_new_tokens=512
    )
    
    # Load test image
    test_image_path = Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return
        
    logger.info(f"Loading test image: {test_image_path}")
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    # Run analysis
    logger.info("Starting VLM analysis...")
    try:
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS RESULTS")
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
        results_file = log_dir / "fixed_llava_results.json"
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
        
        if result.success:
            logger.info("✅ Test completed successfully!")
        else:
            logger.error("❌ Test failed!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    # Cleanup
    logger.info("Cleaning up...")
    client.cleanup()

if __name__ == "__main__":
    main()