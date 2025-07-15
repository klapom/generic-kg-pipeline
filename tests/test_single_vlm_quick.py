#!/usr/bin/env python3
"""
Quick Single VLM Test

Tests just Qwen2.5-VL with a single image to verify functionality.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json

# Setup logging
log_dir = Path("tests/debugging/single_vlm_test")
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

from core.parsers.vlm_integration import MultiVLMIntegration, VLMModelType
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

async def main():
    """Run quick VLM test"""
    
    # Use just one model
    test_models = [VLMModelType.QWEN25_VL_7B]
    
    logger.info("Testing single VLM: Qwen2.5-VL-7B")
    
    # Initialize multi-VLM integration
    multi_vlm = MultiVLMIntegration(enabled_models=test_models)
    
    # Load single test image
    test_image_path = Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return
        
    logger.info(f"Loading test image: {test_image_path}")
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    # Create visual element
    element = VisualElement(
        element_type=VisualElementType.IMAGE,
        source_format=DocumentType.PDF,
        content_hash="test_hash_bmw_front",
        raw_data=image_data,
        page_or_slide=3,
        confidence=1.0,
        file_extension="png"
    )
    element.analysis_metadata = {
        "name": "BMW front view with annotations",
        "expected_features": ["BMW grille", "annotations", "technical labels"]
    }
    
    # Run analysis
    logger.info("Starting VLM analysis...")
    try:
        results_dict = await multi_vlm.analyze_visual_elements_comparative(
            [element],
            max_elements=1
        )
        
        # Log results
        if element.content_hash in results_dict:
            comparison = results_dict[element.content_hash]
            
            logger.info("\n" + "="*60)
            logger.info("ANALYSIS RESULTS")
            logger.info("="*60)
            
            for model, result in comparison.model_results.items():
                logger.info(f"\nModel: {model}")
                logger.info(f"Confidence: {result['confidence']:.2%}")
                logger.info(f"Processing Time: {result['processing_time_seconds']:.2f}s")
                logger.info(f"Description: {result['description'][:200]}...")
                if result.get('extracted_text'):
                    logger.info(f"OCR Text: {result['extracted_text'][:100]}...")
            
            # Save results
            results_file = log_dir / "single_vlm_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "image": element.analysis_metadata['name'],
                    "model_results": comparison.model_results,
                    "consensus": {
                        "score": comparison.consensus_score,
                        "best_model": comparison.best_model
                    },
                    "timestamp": comparison.timestamp
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\nResults saved to: {results_file}")
            logger.info("✅ Test completed successfully!")
            
        else:
            logger.error("No results returned!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())