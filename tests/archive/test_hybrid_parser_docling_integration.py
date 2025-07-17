#!/usr/bin/env python3
"""
Test HybridPDFParser integration with docling final client
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.parsers.implementations.pdf.hybrid_pdf_parser import HybridPDFParser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_legacy_client_configuration():
    """Test HybridPDFParser with legacy client"""
    logger.info("Testing HybridPDFParser with legacy SmolDocling client...")
    
    config = {
        'use_docling_final': False,  # Use legacy client
        'environment': 'development',
        'max_pages': 5,
        'gpu_memory_utilization': 0.2
    }
    
    try:
        parser = HybridPDFParser(config=config, enable_vlm=False)
        
        # Check that it uses the legacy client
        from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
        assert isinstance(parser.smoldocling_client, VLLMSmolDoclingClient)
        
        logger.info("✅ Legacy client configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Legacy client configuration test failed: {e}")
        return False

def test_final_client_configuration():
    """Test HybridPDFParser with final docling client"""
    logger.info("Testing HybridPDFParser with final docling client...")
    
    config = {
        'use_docling_final': True,  # Use final client
        'environment': 'development',
        'max_pages': 5,
        'gpu_memory_utilization': 0.3
    }
    
    try:
        parser = HybridPDFParser(config=config, enable_vlm=False)
        
        # Check that it uses the final client
        from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
        assert isinstance(parser.smoldocling_client, VLLMSmolDoclingFinalClient)
        
        # Check that the client has proper configuration
        assert parser.smoldocling_client.use_docling == True  # Development environment
        assert parser.smoldocling_client.extract_images_directly == True
        assert parser.smoldocling_client.environment == 'development'
        
        logger.info("✅ Final client configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Final client configuration test failed: {e}")
        return False

def test_production_environment():
    """Test HybridPDFParser with production environment configuration"""
    logger.info("Testing HybridPDFParser with production environment...")
    
    config = {
        'use_docling_final': True,
        'environment': 'production',  # Production should be conservative
        'max_pages': 5,
        'gpu_memory_utilization': 0.3
    }
    
    try:
        parser = HybridPDFParser(config=config, enable_vlm=False)
        
        # Check that production settings are applied
        from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
        assert isinstance(parser.smoldocling_client, VLLMSmolDoclingFinalClient)
        assert parser.smoldocling_client.environment == 'production'
        assert parser.smoldocling_client.use_docling == False  # Disabled in production config
        
        logger.info("✅ Production environment test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Production environment test failed: {e}")
        return False

def test_visual_element_handling():
    """Test that visual element handling works correctly"""
    logger.info("Testing visual element handling...")
    
    # Create mock page data with pre-extracted visual elements (simulating final client)
    from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType
    from core.clients.vllm_smoldocling_final import SmolDoclingPage
    
    # Create mock visual element with raw_data
    mock_visual = VisualElement(
        element_type=VisualElementType.IMAGE,
        source_format=DocumentType.PDF,
        content_hash="test_hash_123",
        page_or_slide=1,
        bounding_box=[100, 100, 200, 200],
        raw_data=b"mock_image_bytes",  # Pre-extracted!
        analysis_metadata={"extracted_by": "docling_direct"}
    )
    
    mock_page = SmolDoclingPage(
        page_number=1,
        text="Test page content",
        tables=[],
        images=[],  # Empty because visual_elements is populated
        formulas=[],
        visual_elements=[mock_visual],  # Pre-extracted visual elements
        confidence_score=1.0
    )
    
    try:
        parser = HybridPDFParser(config={'use_docling_final': True}, enable_vlm=False)
        
        # Test the _create_segment_from_smoldocling method
        segment, visual_elements = parser._create_segment_from_smoldocling(mock_page)
        
        # Should have 1 visual element
        assert len(visual_elements) == 1
        
        # Visual element should have raw_data
        assert visual_elements[0].raw_data is not None
        assert visual_elements[0].raw_data == b"mock_image_bytes"
        
        # Should have visual reference in segment
        assert len(segment.visual_references) == 1
        assert segment.visual_references[0] == "test_hash_123"
        
        logger.info("✅ Visual element handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visual element handling test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_backward_compatibility():
    """Test that legacy format still works"""
    logger.info("Testing backward compatibility with legacy format...")
    
    # Create mock page data in legacy format (no visual_elements attribute)
    class MockLegacyPage:
        def __init__(self):
            self.page_number = 1
            self.text = "Legacy page content"
            self.tables = []
            self.formulas = []
            self.confidence_score = 1.0
            # Mock image in legacy format
            self.images = [MockImage()]
    
    class MockImage:
        def __init__(self):
            self.bbox = [50, 50, 150, 150]
            self.caption = "Test image"
            self.content = "<loc_50><loc_50><loc_150><loc_150>image content"
    
    try:
        parser = HybridPDFParser(config={'use_docling_final': False}, enable_vlm=False)
        
        mock_page = MockLegacyPage()
        
        # Test the _create_segment_from_smoldocling method
        segment, visual_elements = parser._create_segment_from_smoldocling(mock_page)
        
        # Should have 1 visual element (created from legacy format)
        assert len(visual_elements) == 1
        
        # Visual element should NOT have raw_data (will be extracted later)
        assert visual_elements[0].raw_data is None
        
        # Should have correct bbox
        assert visual_elements[0].bounding_box == [50, 50, 150, 150]
        
        logger.info("✅ Backward compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backward compatibility test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all integration tests"""
    logger.info("Starting HybridPDFParser docling integration tests...")
    
    tests = [
        ("Legacy Client Configuration", test_legacy_client_configuration),
        ("Final Client Configuration", test_final_client_configuration),
        ("Production Environment", test_production_environment),
        ("Visual Element Handling", test_visual_element_handling),
        ("Backward Compatibility", test_backward_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed! Phase 3 is complete.")
        return 0
    else:
        logger.error("⚠️  Some integration tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())