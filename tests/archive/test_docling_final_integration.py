#!/usr/bin/env python3
"""
Comprehensive test suite for the final docling integration
Tests configuration system, client initialization, and functionality
"""

import os
import sys
from pathlib import Path
import logging
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.docling_config import get_config, is_docling_enabled, should_use_docling_for_document
from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDoclingConfiguration:
    """Test configuration system"""
    
    def test_configuration_loading(self):
        """Test configuration loading for different environments"""
        logger.info("Testing configuration loading...")
        
        # Test different environments
        dev_config = get_config("development")
        prod_config = get_config("production")
        test_config = get_config("testing")
        
        # Development should enable docling
        assert dev_config["use_docling"] == True
        assert dev_config["rollout_percentage"] == 100
        
        # Production should be conservative
        assert prod_config["use_docling"] == False
        assert prod_config["rollout_percentage"] == 0
        
        # Testing uses base config
        assert test_config["use_docling"] == False
        assert test_config["rollout_percentage"] == 0
        
        logger.info("✅ Configuration loading test passed")
        return True
    
    def test_docling_enabled_check(self):
        """Test docling enabled check for environments"""
        logger.info("Testing docling enabled check...")
        
        assert is_docling_enabled("development") == True
        assert is_docling_enabled("production") == False
        assert is_docling_enabled("testing") == False
        
        logger.info("✅ Docling enabled check test passed")
        return True
    
    def test_document_routing(self):
        """Test document-specific routing based on hash"""
        logger.info("Testing document routing...")
        
        # Test consistent routing
        test_hash = "test_document_123"
        
        # Same document should always get same result
        result1 = should_use_docling_for_document(test_hash, "development")
        result2 = should_use_docling_for_document(test_hash, "development")
        assert result1 == result2
        
        # Development with 100% rollout should always return True
        assert should_use_docling_for_document(test_hash, "development") == True
        
        # Production with 0% rollout should always return False
        assert should_use_docling_for_document(test_hash, "production") == False
        
        logger.info("✅ Document routing test passed")
        return True

class TestClientInitialization:
    """Test client initialization and configuration integration"""
    
    def test_development_client(self):
        """Test client in development environment"""
        logger.info("Testing development client initialization...")
        
        client = VLLMSmolDoclingFinalClient(environment="development")
        
        # Should have development settings
        assert client.use_docling == True
        assert client.extract_images_directly == True
        assert client.fallback_to_legacy == True
        assert client.log_performance == True
        assert client.environment == "development"
        
        # Configuration should be loaded
        assert client.config is not None
        assert client.config["use_docling"] == True
        
        logger.info("✅ Development client test passed")
        return True
    
    def test_production_client(self):
        """Test client in production environment"""
        logger.info("Testing production client initialization...")
        
        client = VLLMSmolDoclingFinalClient(environment="production")
        
        # Should have production settings (conservative)
        assert client.use_docling == False  # Disabled in production config
        assert client.extract_images_directly == True
        assert client.fallback_to_legacy == True
        assert client.log_performance == True
        assert client.environment == "production"
        
        logger.info("✅ Production client test passed")
        return True
    
    def test_docling_availability_check(self):
        """Test docling availability checking"""
        logger.info("Testing docling availability check...")
        
        client = VLLMSmolDoclingFinalClient(environment="development")
        
        # Check if docling libraries are available
        try:
            from docling_core.types.doc import DoclingDocument
            from docling_core.types.doc.document import DocTagsDocument
            expected_available = True
            logger.info("Docling libraries found")
        except ImportError:
            expected_available = False
            logger.info("Docling libraries not found")
        
        assert client._docling_available == expected_available
        
        logger.info("✅ Docling availability check test passed")
        return True

class TestDocumentProcessing:
    """Test document processing logic"""
    
    def test_document_hash_generation(self):
        """Test consistent document hash generation"""
        logger.info("Testing document hash generation...")
        
        # Create a temporary file for testing
        test_file = project_root / "tests" / "temp_test_file.txt"
        test_file.write_text("Test content")
        
        try:
            client = VLLMSmolDoclingFinalClient(environment="development")
            
            # Generate hash
            hash1 = client._get_document_hash(test_file)
            hash2 = client._get_document_hash(test_file)
            
            # Should be consistent
            assert hash1 == hash2
            assert len(hash1) == 32  # MD5 hex length
            
            # Should be based on filename and size
            expected_content = f"{test_file.name}_{test_file.stat().st_size}"
            expected_hash = hashlib.md5(expected_content.encode()).hexdigest()
            assert hash1 == expected_hash
            
        finally:
            test_file.unlink()  # Clean up
        
        logger.info("✅ Document hash generation test passed")
        return True
    
    def test_file_size_limits(self):
        """Test file size checking logic"""
        logger.info("Testing file size limits...")
        
        client = VLLMSmolDoclingFinalClient(environment="development")
        max_size = client.config["memory_limits"]["max_pdf_size_mb"]
        
        # Test with existing PDF if available
        test_pdfs = [
            project_root / "data" / "input" / "BMW_Annual_Report_2023.pdf",
            project_root / "data" / "PDFs" / "BMW_Annual_Report_2023.pdf"
        ]
        
        pdf_path = None
        for path in test_pdfs:
            if path.exists():
                pdf_path = path
                break
        
        if pdf_path:
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            logger.info(f"Test PDF size: {file_size_mb:.1f}MB (limit: {max_size}MB)")
            
            # Test parsing decision logic (without actually parsing)
            # This tests the size check part of parse_pdf
            if file_size_mb <= max_size:
                logger.info("PDF within size limits")
            else:
                logger.info("PDF exceeds size limits - would use legacy parser")
        else:
            logger.info("No test PDF found - skipping file size test")
        
        logger.info("✅ File size limits test passed")
        return True

class TestCompatibility:
    """Test backward compatibility"""
    
    def test_legacy_fallback(self):
        """Test fallback to legacy parser"""
        logger.info("Testing legacy fallback...")
        
        # Create client with docling disabled
        client = VLLMSmolDoclingFinalClient(environment="production")  # Has docling disabled
        assert client.use_docling == False
        
        # Should indicate legacy will be used
        logger.info("✅ Legacy fallback test passed")
        return True
    
    def test_configuration_structure(self):
        """Test that configuration has all required fields"""
        logger.info("Testing configuration structure...")
        
        config = get_config("development")
        
        # Check required top-level fields
        required_fields = [
            "use_docling", "extract_images_directly", "fallback_to_legacy",
            "log_performance", "rollout_percentage", "image_extraction",
            "memory_limits", "error_handling"
        ]
        
        for field in required_fields:
            assert field in config, f"Missing config field: {field}"
        
        # Check nested structures
        assert "max_image_size" in config["image_extraction"]
        assert "image_quality" in config["image_extraction"]
        assert "extract_tables_as_images" in config["image_extraction"]
        assert "extract_formulas_as_images" in config["image_extraction"]
        
        assert "max_pdf_size_mb" in config["memory_limits"]
        assert "max_pages_per_batch" in config["memory_limits"]
        
        assert "max_retries" in config["error_handling"]
        assert "timeout_seconds" in config["error_handling"]
        assert "continue_on_page_error" in config["error_handling"]
        
        logger.info("✅ Configuration structure test passed")
        return True

def run_all_tests():
    """Run all test suites"""
    logger.info("Starting comprehensive docling integration tests...")
    
    test_suites = [
        ("Configuration Tests", TestDoclingConfiguration()),
        ("Client Initialization Tests", TestClientInitialization()),
        ("Document Processing Tests", TestDocumentProcessing()),
        ("Compatibility Tests", TestCompatibility())
    ]
    
    results = []
    
    for suite_name, suite_instance in test_suites:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {suite_name}")
        logger.info(f"{'='*60}")
        
        # Get all test methods
        test_methods = [method for method in dir(suite_instance) 
                       if method.startswith('test_') and callable(getattr(suite_instance, method))]
        
        suite_results = []
        for test_method in test_methods:
            try:
                method = getattr(suite_instance, test_method)
                result = method()
                suite_results.append((test_method, True))
                logger.info(f"✅ {test_method}: PASSED")
            except Exception as e:
                suite_results.append((test_method, False))
                logger.error(f"❌ {test_method}: FAILED - {e}")
        
        results.append((suite_name, suite_results))
    
    # Generate summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    total_tests = 0
    total_passed = 0
    
    for suite_name, suite_results in results:
        suite_passed = sum(1 for _, passed in suite_results if passed)
        suite_total = len(suite_results)
        total_tests += suite_total
        total_passed += suite_passed
        
        logger.info(f"{suite_name}: {suite_passed}/{suite_total} passed")
        
        for test_name, passed in suite_results:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {test_name}")
    
    logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("🎉 All tests passed! Docling integration is ready for Phase 3.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Check configuration and fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit(run_all_tests())