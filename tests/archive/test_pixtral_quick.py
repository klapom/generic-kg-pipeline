#!/usr/bin/env python3
"""
Quick Pixtral Availability Test

Tests if Pixtral can be loaded without running full inference.
"""

import logging
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path("tests/debugging/pixtral_quick_test")
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

def test_pixtral_processor():
    """Test if Pixtral processor can be loaded"""
    logger.info("Testing Pixtral processor availability...")
    
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        logger.info("✅ Pixtral processor loaded successfully")
        logger.info(f"Processor type: {type(processor)}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load Pixtral processor: {e}")
        return False

def test_pixtral_model_info():
    """Get model information without downloading"""
    logger.info("Getting Pixtral model information...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("mistral-community/pixtral-12b")
        logger.info("✅ Pixtral config loaded successfully")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Architecture: {config.architectures}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to get Pixtral config: {e}")
        return False

def main():
    """Run quick Pixtral tests"""
    
    logger.info("Starting quick Pixtral availability test...")
    
    # Test 1: Model config
    config_ok = test_pixtral_model_info()
    
    # Test 2: Processor
    processor_ok = test_pixtral_processor()
    
    # Summary
    if config_ok and processor_ok:
        logger.info("✅ Pixtral is available and ready for integration!")
        logger.info("Note: Model weights will be downloaded on first use.")
    else:
        logger.error("❌ Pixtral has issues - check network connection or model availability")

if __name__ == "__main__":
    main()