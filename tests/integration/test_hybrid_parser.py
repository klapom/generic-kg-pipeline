#!/usr/bin/env python3
"""
Test the hybrid PDF parser with BMW document
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.parsers.hybrid_pdf_parser import HybridPDFParser
from core.parsers import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce vLLM/urllib3 noise
logging.getLogger('vllm').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)

def test_hybrid_parser():
    """Test hybrid parser on BMW PDF"""
    
    # Create document
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    # For HybridPDFParser, we need a simple document object with file_path
    # Create a simple Document-like object
    class SimpleDocument:
        def __init__(self, file_path):
            self.file_path = Path(file_path)
    
    document = SimpleDocument(pdf_path)
    
    # Create parser with config
    config = {
        'max_pages': 5,
        'gpu_memory_utilization': 0.2,
        'prefer_pdfplumber': False,  # Use PyPDF2 for now
        'fallback_confidence_threshold': 0.8
    }
    
    logger.info("🚀 Testing Hybrid PDF Parser...")
    parser = HybridPDFParser(config)
    
    # Load SmolDocling model
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(parser.smoldocling_client.model_id)
    
    # Parse document
    try:
        segments = parser.parse(document)
        
        logger.info(f"\n✅ Parsing complete! Got {len(segments)} segments")
        
        # Analyze results
        fallback_pages = []
        smol_pages = []
        
        for seg in segments:
            page_num = seg.metadata.get('page_number', 0)
            parser_used = seg.metadata.get('parser_used', seg.metadata.get('parser', 'unknown'))
            
            if parser_used == 'fallback':
                fallback_pages.append(page_num)
            else:
                smol_pages.append(page_num)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 Page {page_num}:")
            logger.info(f"  Parser: {parser_used}")
            logger.info(f"  Content length: {len(seg.content)} chars")
            logger.info(f"  Content preview: {seg.content[:100]}..." if seg.content else "  No content")
            
            if parser_used == 'fallback':
                logger.info(f"  Fallback extractor: {seg.metadata.get('fallback_extractor', 'N/A')}")
                logger.info(f"  Tables extracted: {seg.metadata.get('tables_extracted', 0)}")
                logger.info(f"  Lists extracted: {seg.metadata.get('lists_extracted', 0)}")
            else:
                logger.info(f"  Tables: {seg.metadata.get('table_count', 0)}")
                logger.info(f"  Images: {seg.metadata.get('image_count', 0)}")
        
        # Special check for page 2
        page2_segments = [s for s in segments if s.metadata.get('page_number') == 2]
        if page2_segments:
            page2 = page2_segments[0]
            logger.info(f"\n{'='*80}")
            logger.info("🔍 PAGE 2 DETAILED ANALYSIS:")
            logger.info(f"{'='*80}")
            logger.info(f"Parser used: {page2.metadata.get('parser_used', page2.metadata.get('parser'))}")
            logger.info(f"Content extracted: {len(page2.content)} chars")
            
            # Check if we got the expected content
            expected_keywords = ['Motorisierung', 'Weltpremiere', '320d', 'Highlights']
            found_keywords = [kw for kw in expected_keywords if kw in page2.content]
            
            logger.info(f"Found keywords: {found_keywords}")
            
            if len(found_keywords) >= 2:
                logger.info("✅ SUCCESS! Page 2 content successfully extracted with fallback!")
            else:
                logger.warning("⚠️  Page 2 extraction might be incomplete")
                logger.info("Content sample:")
                logger.info(page2.content[:500])
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("📊 FINAL SUMMARY:")
        logger.info(f"{'='*80}")
        logger.info(f"Total segments: {len(segments)}")
        logger.info(f"SmolDocling pages: {len(smol_pages)} - {smol_pages}")
        logger.info(f"Fallback pages: {len(fallback_pages)} - {fallback_pages}")
        
        # Save page 2 content for inspection
        if page2_segments:
            with open('data/output/page2_fallback_content.txt', 'w', encoding='utf-8') as f:
                f.write(page2_segments[0].content)
            logger.info(f"\nPage 2 content saved to: data/output/page2_fallback_content.txt")
        
        return segments
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    segments = test_hybrid_parser()