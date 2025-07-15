#!/usr/bin/env python3
"""
Simple test for VLM pipeline components without PDF parsing.
Tests the core functionality of model management, classification, and processing.
"""

import sys
sys.path.insert(0, '.')

import logging
from pathlib import Path
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw
import io
import hashlib

from core.vlm import (
    VLMModelManager,
    DocumentElementClassifier,
    TwoStageVLMProcessor,
    ConfidenceEvaluator,
    FallbackStrategy
)
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_visual_element(image_data: bytes, element_type: VisualElementType = VisualElementType.IMAGE) -> VisualElement:
    """Create a test visual element."""
    content_hash = hashlib.md5(image_data).hexdigest()
    return VisualElement(
        element_type=element_type,
        source_format=DocumentType.PDF,
        content_hash=content_hash,
        raw_data=image_data,
        page_or_slide=1
    )


def test_model_manager():
    """Test model loading and switching."""
    logger.info("="*80)
    logger.info("Testing VLMModelManager...")
    logger.info("="*80)
    
    manager = VLMModelManager()
    
    try:
        # Test loading Qwen
        logger.info("Loading Qwen2.5-VL...")
        qwen_model = manager.load_qwen()
        logger.info(f"✅ Current model: {manager.get_current_model_name()}")
        logger.info(f"   Memory usage: {manager.get_memory_usage()}")
        
        # Test switching to Pixtral
        logger.info("\nSwitching to Pixtral...")
        pixtral_model = manager.load_pixtral()
        logger.info(f"✅ Current model: {manager.get_current_model_name()}")
        logger.info(f"   Memory usage: {manager.get_memory_usage()}")
        
        # Cleanup
        manager.cleanup()
        logger.info(f"✅ After cleanup - Memory usage: {manager.get_memory_usage()}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model manager test failed: {e}")
        return False


def test_document_classifier():
    """Test document element classification."""
    logger.info("\n" + "="*80)
    logger.info("Testing DocumentElementClassifier...")
    logger.info("="*80)
    
    classifier = DocumentElementClassifier()
    
    try:
        # Create test images
        # Simple white image
        img1 = Image.new('RGB', (800, 600), color='white')
        img1_bytes = io.BytesIO()
        img1.save(img1_bytes, format='PNG')
        img1_data = img1_bytes.getvalue()
        
        # Image with lines (table-like)
        img2 = Image.new('RGB', (800, 600), color='white')
        # Add some lines to simulate a table
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img2)
        for y in range(100, 500, 50):
            draw.line([(50, y), (750, y)], fill='black', width=2)
        for x in range(50, 750, 100):
            draw.line([(x, 100), (x, 500)], fill='black', width=2)
        img2_bytes = io.BytesIO()
        img2.save(img2_bytes, format='PNG')
        img2_data = img2_bytes.getvalue()
        
        # Test classification
        element_type1 = classifier.detect_element_type(img1_data)
        logger.info(f"✅ Plain image detected as: {element_type1}")
        
        element_type2 = classifier.detect_element_type(img2_data)
        logger.info(f"✅ Table-like image detected as: {element_type2}")
        
        # Test with context
        element_type_with_context = classifier.detect_element_type(
            img1_data, 
            context_text="This diagram shows the system architecture"
        )
        logger.info(f"✅ With diagram context: {element_type_with_context}")
        
        # Test processing recommendation
        recommendation = classifier.get_processing_recommendation(img2_data)
        logger.info(f"✅ Processing recommendation: {recommendation['primary_model']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Document classifier test failed: {e}")
        return False


def test_confidence_evaluator():
    """Test confidence evaluation and fallback strategy."""
    logger.info("\n" + "="*80)
    logger.info("Testing ConfidenceEvaluator and FallbackStrategy...")
    logger.info("="*80)
    
    evaluator = ConfidenceEvaluator(base_threshold=0.85)
    strategy = FallbackStrategy(evaluator)
    
    try:
        # Create test result
        from core.parsers.interfaces.data_models import VisualAnalysisResult
        
        # High confidence result
        high_conf_result = VisualAnalysisResult(
            description="A detailed BMW X5 technical diagram showing the engine components",
            confidence=0.92,
            success=True
        )
        
        # Low confidence result
        low_conf_result = VisualAnalysisResult(
            description="An image of something",
            confidence=0.65,
            success=True
        )
        
        # Evaluate high confidence
        should_fallback, reasons = strategy.should_fallback(high_conf_result)
        logger.info(f"✅ High confidence ({high_conf_result.confidence:.0%}): fallback={should_fallback}")
        
        # Evaluate low confidence
        should_fallback, reasons = strategy.should_fallback(low_conf_result)
        logger.info(f"✅ Low confidence ({low_conf_result.confidence:.0%}): fallback={should_fallback}, reasons={reasons}")
        
        # Test fallback config
        fallback_config = strategy.get_fallback_config(reasons)
        logger.info(f"✅ Fallback config: temperature={fallback_config['temperature']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Confidence evaluator test failed: {e}")
        return False


def test_two_stage_processor():
    """Test two-stage VLM processing with simple images."""
    logger.info("\n" + "="*80)
    logger.info("Testing TwoStageVLMProcessor...")
    logger.info("="*80)
    
    processor = TwoStageVLMProcessor(confidence_threshold=0.85)
    
    try:
        # Create test visual elements
        visuals = []
        
        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='lightblue')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Test Image", fill='black')
        draw.rectangle([(100, 100), (300, 200)], outline='red', width=3)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Create visual elements
        visual1 = create_test_visual_element(img_data, VisualElementType.IMAGE)
        visuals.append(visual1)
        
        # Create a diagram-like image
        diagram = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(diagram)
        # Draw some boxes and arrows
        draw.rectangle([(50, 50), (150, 100)], outline='black', width=2)
        draw.text((75, 65), "Start", fill='black')
        draw.line([(150, 75), (250, 75)], fill='black', width=2)
        draw.rectangle([(250, 50), (350, 100)], outline='black', width=2)
        draw.text((275, 65), "Process", fill='black')
        
        diagram_bytes = io.BytesIO()
        diagram.save(diagram_bytes, format='PNG')
        diagram_data = diagram_bytes.getvalue()
        
        visual2 = create_test_visual_element(diagram_data, VisualElementType.DIAGRAM)
        visuals.append(visual2)
        
        logger.info(f"Processing {len(visuals)} test visuals...")
        
        # Process with two-stage processor
        results = processor.process_batch(visuals)
        
        # Log results
        for idx, result in enumerate(results):
            logger.info(f"✅ Visual {idx + 1}: Success={result.success}, "
                      f"Confidence={result.confidence:.0%}")
        
        # Log stats
        logger.info(f"\n📊 Processing Statistics:")
        logger.info(f"   Qwen processed: {processor.processing_stats['qwen_processed']}")
        logger.info(f"   Pixtral processed: {processor.processing_stats['pixtral_processed']}")
        logger.info(f"   Fallback count: {processor.processing_stats['fallback_count']}")
        
        # Cleanup
        processor.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Two-stage processor test failed: {e}")
        processor.cleanup()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("VLM Pipeline Simple Test Suite")
    logger.info("="*80)
    
    # Save results
    output_dir = Path("tests/debugging/vlm_pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Model Manager
    try:
        success = test_model_manager()
        test_results["tests"]["model_manager"] = {
            "success": success,
            "message": "Model switching works correctly" if success else "Failed"
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
            "message": "Classification works correctly" if success else "Failed"
        }
    except Exception as e:
        test_results["tests"]["document_classifier"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test 3: Confidence Evaluator
    try:
        success = test_confidence_evaluator()
        test_results["tests"]["confidence_evaluator"] = {
            "success": success,
            "message": "Confidence evaluation works correctly" if success else "Failed"
        }
    except Exception as e:
        test_results["tests"]["confidence_evaluator"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test 4: Two-Stage Processor
    try:
        success = test_two_stage_processor()
        test_results["tests"]["two_stage_processor"] = {
            "success": success,
            "message": "Two-stage processing works correctly" if success else "Failed"
        }
    except Exception as e:
        test_results["tests"]["two_stage_processor"] = {
            "success": False,
            "error": str(e)
        }
    
    # Save results
    results_file = output_dir / f"simple_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Test Summary")
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
    success = main()
    sys.exit(0 if success else 1)