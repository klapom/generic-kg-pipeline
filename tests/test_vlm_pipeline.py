#!/usr/bin/env python3
"""
Test the complete VLM pipeline with BMW X5 document.
Tests two-stage processing, batch processing, and confidence-based fallback.
"""

import sys
sys.path.insert(0, '.')

import asyncio
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from core.vlm import (
    VLMModelManager,
    DocumentElementClassifier,
    TwoStageVLMProcessor,
    BatchDocumentProcessor,
    BatchProcessingConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_model_manager():
    """Test model loading and switching."""
    logger.info("="*80)
    logger.info("Testing VLMModelManager...")
    logger.info("="*80)
    
    manager = VLMModelManager()
    
    # Test loading Qwen
    logger.info("Loading Qwen2.5-VL...")
    qwen_model = manager.load_qwen()
    logger.info(f"Current model: {manager.get_current_model_name()}")
    logger.info(f"Memory usage: {manager.get_memory_usage()}")
    
    # Test switching to Pixtral
    logger.info("\nSwitching to Pixtral...")
    pixtral_model = manager.load_pixtral()
    logger.info(f"Current model: {manager.get_current_model_name()}")
    logger.info(f"Memory usage: {manager.get_memory_usage()}")
    
    # Cleanup
    manager.cleanup()
    logger.info(f"After cleanup - Memory usage: {manager.get_memory_usage()}")
    
    return True


def test_document_classifier():
    """Test document element classification."""
    logger.info("\n" + "="*80)
    logger.info("Testing DocumentElementClassifier...")
    logger.info("="*80)
    
    classifier = DocumentElementClassifier()
    
    # Test with sample image (create a simple test image)
    from PIL import Image
    import io
    
    # Create a test image
    img = Image.new('RGB', (800, 600), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    
    # Test classification
    element_type = classifier.detect_element_type(img_data)
    logger.info(f"Detected element type: {element_type}")
    
    # Test with context
    element_type_with_context = classifier.detect_element_type(
        img_data, 
        context_text="This diagram shows the system architecture"
    )
    logger.info(f"Detected with context: {element_type_with_context}")
    
    # Test processing recommendation
    recommendation = classifier.get_processing_recommendation(img_data)
    logger.info(f"Processing recommendation: {json.dumps(recommendation, indent=2)}")
    
    return True


async def test_two_stage_processor():
    """Test two-stage VLM processing."""
    logger.info("\n" + "="*80)
    logger.info("Testing TwoStageVLMProcessor...")
    logger.info("="*80)
    
    # Parse BMW document to get visual elements
    from core.parsers.hybrid_pdf_parser import HybridPDFParser
    
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    if not pdf_path.exists():
        logger.warning(f"Test PDF not found: {pdf_path}")
        return False
    
    parser = HybridPDFParser()
    processor = TwoStageVLMProcessor(confidence_threshold=0.85)
    
    try:
        # Parse document
        logger.info(f"Parsing {pdf_path.name}...")
        parsed_doc = await parser.parse(pdf_path)
        
        # Collect visuals (limit to first few for testing)
        visuals = parsed_doc.visual_elements[:5]  # First 5 visuals
        
        logger.info(f"Found {len(parsed_doc.visual_elements)} total visuals, using first {len(visuals)} for testing")
        
        if visuals:
            # Process with two-stage processor
            logger.info("Processing visuals with two-stage strategy...")
            results = processor.process_batch(visuals[:5])  # Test with first 5 visuals
            
            # Log results
            for idx, result in enumerate(results):
                logger.info(f"Visual {idx + 1}: Success={result.success}, "
                          f"Confidence={result.confidence:.0%}")
        
        # Cleanup
        processor.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in two-stage processing test: {e}")
        processor.cleanup()
        return False


async def test_batch_processor():
    """Test batch document processing."""
    logger.info("\n" + "="*80)
    logger.info("Testing BatchDocumentProcessor...")
    logger.info("="*80)
    
    # Configure batch processing
    config = BatchProcessingConfig(
        max_batch_size=8,
        max_memory_gb=14.0,
        enable_parallel_extraction=True,
        confidence_threshold=0.85
    )
    
    processor = BatchDocumentProcessor(config)
    
    # Test with BMW document
    pdf_paths = [Path("data/input/Preview_BMW_X5_G05.pdf")]
    
    # Add more PDFs if available
    data_dir = Path("data/input")
    if data_dir.exists():
        additional_pdfs = list(data_dir.glob("*.pdf"))[:2]  # Max 2 additional
        pdf_paths.extend(additional_pdfs)
    
    logger.info(f"Processing {len(pdf_paths)} documents in batch")
    
    try:
        # Process batch
        results = await processor.process_documents(pdf_paths)
        
        # Log results
        for result in results:
            logger.info(f"\nDocument: {result.document_path.name}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Visuals: {result.visual_count}")
            logger.info(f"  Time: {result.processing_time:.1f}s")
            if result.error:
                logger.info(f"  Error: {result.error}")
        
        # Cleanup
        processor.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Error in batch processing test: {e}")
        processor.cleanup()
        return False


async def test_complete_pipeline():
    """Test the complete VLM pipeline end-to-end."""
    logger.info("\n" + "="*80)
    logger.info("Testing Complete VLM Pipeline...")
    logger.info("="*80)
    
    # Save results
    output_dir = Path("tests/debugging/vlm_pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all tests
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Model Manager
    try:
        success = await test_model_manager()
        test_results["tests"]["model_manager"] = {
            "success": success,
            "message": "Model switching works correctly"
        }
    except Exception as e:
        test_results["tests"]["model_manager"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test 2: Document Classifier
    try:
        success = test_document_classifier()
        test_results["tests"]["document_classifier"] = {
            "success": success,
            "message": "Classification works correctly"
        }
    except Exception as e:
        test_results["tests"]["document_classifier"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test 3: Two-Stage Processor
    try:
        success = await test_two_stage_processor()
        test_results["tests"]["two_stage_processor"] = {
            "success": success,
            "message": "Two-stage processing works correctly"
        }
    except Exception as e:
        test_results["tests"]["two_stage_processor"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test 4: Batch Processor
    try:
        success = await test_batch_processor()
        test_results["tests"]["batch_processor"] = {
            "success": success,
            "message": "Batch processing works correctly"
        }
    except Exception as e:
        test_results["tests"]["batch_processor"] = {
            "success": False,
            "error": str(e)
        }
    
    # Save results
    results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VLM Pipeline Test Summary")
    logger.info("="*80)
    
    all_passed = all(test["success"] for test in test_results["tests"].values())
    
    for test_name, result in test_results["tests"].items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result["success"] and "error" in result:
            logger.info(f"  Error: {result['error']}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    # Run complete pipeline test
    asyncio.run(test_complete_pipeline())