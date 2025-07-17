#!/usr/bin/env python3
"""
Test Available VLMs

Tests VLMs that are already downloaded:
- Qwen2.5-VL-7B
- LLaVA-1.6-Mistral-7B (already downloaded)
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/available_vlms_test")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'comparison_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, '.')

from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.parsers.interfaces.data_models import VisualElementType

def test_vlm(client, image_data: bytes, model_name: str) -> Dict[str, Any]:
    """Test a single VLM client"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    try:
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        logger.info(f"Success: {result.success}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Processing Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"Description Preview: {result.description[:150]}...")
        if result.ocr_text:
            logger.info(f"OCR Text: {result.ocr_text}")
        
        return {
            "model": model_name,
            "success": result.success,
            "confidence": result.confidence,
            "processing_time": result.processing_time_seconds,
            "description": result.description,
            "ocr_text": result.ocr_text,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Failed to test {model_name}: {e}")
        return {
            "model": model_name,
            "success": False,
            "error_message": str(e)
        }
    finally:
        # Cleanup
        if hasattr(client, 'cleanup'):
            client.cleanup()

def main():
    """Run available VLMs comparison"""
    
    # Test with two different images
    test_images = [
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
            "name": "BMW front view"
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
            "name": "BMW interior"
        }
    ]
    
    all_results = []
    
    for img_info in test_images:
        if not img_info["path"].exists():
            logger.warning(f"Test image not found: {img_info['path']}")
            continue
            
        logger.info(f"\n{'#'*60}")
        logger.info(f"Testing with image: {img_info['name']}")
        logger.info('#'*60)
        
        with open(img_info["path"], 'rb') as f:
            image_data = f.read()
        
        results = []
        
        # Test Qwen2.5-VL
        logger.info("\nInitializing Qwen2.5-VL-7B...")
        qwen_client = TransformersQwen25VLClient(
            temperature=0.1,
            max_new_tokens=512
        )
        result = test_vlm(qwen_client, image_data, "Qwen2.5-VL-7B")
        result["image"] = img_info["name"]
        results.append(result)
        
        # Test LLaVA
        logger.info("\nInitializing LLaVA-1.6-Mistral-7B...")
        llava_client = TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.1,
            max_new_tokens=512
        )
        result = test_vlm(llava_client, image_data, "LLaVA-1.6-Mistral-7B")
        result["image"] = img_info["name"]
        results.append(result)
        
        all_results.extend(results)
    
    # Save JSON results
    results_file = log_dir / "available_vlms_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    
    # Group by image
    for img_name in ["BMW front view", "BMW interior"]:
        img_results = [r for r in all_results if r.get("image") == img_name]
        if not img_results:
            continue
            
        logger.info(f"\n📸 Image: {img_name}")
        logger.info("-"*40)
        
        for result in img_results:
            if result.get('success'):
                logger.info(f"\n✅ {result['model']}")
                logger.info(f"   Confidence: {result.get('confidence', 0):.1%}")
                logger.info(f"   Processing: {result.get('processing_time', 0):.2f}s")
                desc = result.get('description', '')[:100].replace('\n', ' ')
                logger.info(f"   Description: {desc}...")
            else:
                logger.info(f"\n❌ {result['model']}")
                logger.info(f"   Error: {result.get('error_message', 'Unknown')}")
    
    # Performance comparison
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*80)
    
    for model in ["Qwen2.5-VL-7B", "LLaVA-1.6-Mistral-7B"]:
        model_results = [r for r in all_results if r.get("model") == model and r.get("success")]
        if model_results:
            avg_time = sum(r["processing_time"] for r in model_results) / len(model_results)
            avg_conf = sum(r["confidence"] for r in model_results) / len(model_results)
            logger.info(f"\n{model}:")
            logger.info(f"  Average Processing Time: {avg_time:.2f}s")
            logger.info(f"  Average Confidence: {avg_conf:.1%}")
    
    logger.info("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()