#!/usr/bin/env python3
"""Test Qwen2.5-VL Transformers Client"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
log_dir = Path("tests/debugging/qwen25vl_transformers_client")
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

# Import our modules
import sys
sys.path.insert(0, '.')

from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.parsers.interfaces.data_models import VisualElementType

async def test_client():
    """Test Transformers-based Qwen2.5-VL client"""
    
    try:
        # Initialize client
        logger.info("Initializing TransformersQwen25VLClient...")
        client = TransformersQwen25VLClient()
        
        # Check health
        health = client.health_check()
        logger.info(f"Health check: {health}")
        
        # Test images
        test_images = [
            {
                "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
                "name": "BMW front view with annotations",
                "element_type": VisualElementType.IMAGE
            },
            {
                "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
                "name": "BMW interior dashboard",
                "element_type": VisualElementType.IMAGE
            }
        ]
        
        results = []
        
        for img_info in test_images:
            if not img_info["path"].exists():
                logger.warning(f"Image not found: {img_info['path']}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {img_info['name']}")
            logger.info(f"{'='*60}")
            
            # Load image
            with open(img_info["path"], 'rb') as f:
                image_bytes = f.read()
            
            # Test different analysis modes
            for analysis_focus in ["comprehensive", "description"]:
                logger.info(f"\nAnalysis focus: {analysis_focus}")
                
                try:
                    result = client.analyze_visual(
                        image_data=image_bytes,
                        element_type=img_info["element_type"],
                        analysis_focus=analysis_focus
                    )
                    
                    logger.info(f"Success: {result.success}")
                    logger.info(f"Confidence: {result.confidence:.2%}")
                    logger.info(f"Description: {result.description[:200]}...")
                    if result.ocr_text:
                        logger.info(f"OCR Text: {result.ocr_text[:100]}...")
                    
                    results.append({
                        "image": img_info["name"],
                        "analysis_focus": analysis_focus,
                        "success": result.success,
                        "confidence": result.confidence,
                        "description": result.description,
                        "ocr_text": result.ocr_text,
                        "processing_time": result.processing_time_seconds
                    })
                    
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    results.append({
                        "image": img_info["name"],
                        "analysis_focus": analysis_focus,
                        "success": False,
                        "error": str(e)
                    })
        
        # Save results
        results_file = log_dir / "client_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Cleanup
        client.cleanup()
        logger.info("Client cleaned up")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_client())