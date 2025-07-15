#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
from core.clients.transformers_pixtral_client import TransformersPixtralClient
import logging
import time

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test with BMW image
test_image = Path('tests/debugging/key_pages_vlm_analysis/extracted_visuals/page_001_full.png')

if test_image.exists():
    logger.info('🔧 Testing Pixtral with all fixes...')
    
    client = TransformersPixtralClient(
        temperature=0.3,
        max_new_tokens=512,
        load_in_8bit=True  # Use 8-bit to avoid bfloat16 issues
    )
    
    logger.info('📸 Testing with BMW title page...')
    
    with open(test_image, 'rb') as f:
        image_data = f.read()
    
    start = time.time()
    result = client.analyze_visual(
        image_data=image_data,
        analysis_focus='comprehensive'
    )
    elapsed = time.time() - start
    
    logger.info(f'\n✅ Analysis completed in {elapsed:.1f}s')
    logger.info(f'\n📊 Results:')
    logger.info(f'Success: {result.success}')
    if result.success:
        logger.info(f'Confidence: {result.confidence:.0%}')
        logger.info(f'\n📝 Description:\n{result.description}')
        
        if result.ocr_text:
            logger.info(f'\n🔤 OCR Text:\n{result.ocr_text}')
        
        if result.extracted_data:
            logger.info(f'\n📊 Extracted Data:\n{result.extracted_data}')
    else:
        logger.error(f'Error: {result.error_message}')
    
    # Cleanup
    client.cleanup()
else:
    logger.error(f'Test image not found: {test_image}')