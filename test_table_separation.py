#!/usr/bin/env python3
"""
Test table and text separation functionality
"""

import logging
from pathlib import Path
from core.parsers.hybrid_pdf_parser import HybridPDFParser
from core.parsers.table_text_separator import TableTextSeparator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_table_separation():
    """Test table/text separation on BMW PDF"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🔬 TESTING TABLE/TEXT SEPARATION")
    logger.info("="*80)
    
    # Configure parser with table separation enabled
    config = {
        'max_pages': 6,  # Test first 6 pages
        'gpu_memory_utilization': 0.2,
        'prefer_pdfplumber': True,
        'fallback_confidence_threshold': 0.8,
        'separate_tables': True  # Enable table separation
    }
    
    parser = HybridPDFParser(config)
    
    # Load model
    logger.info("\n📦 Loading SmolDocling model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(parser.smoldocling_client.model_id)
    
    # Process document
    logger.info("\n📄 Processing BMW PDF with table separation...")
    
    try:
        document = parser.parse(pdf_path)
        segments = document.segments
        
        logger.info(f"\n✅ Processing complete. Total segments: {len(segments)}")
        
        # Analyze pages with tables
        for segment in segments:
            page_num = segment.metadata.get('page_number', 0)
            
            # Check pages 2 and 6 specifically
            if page_num in [2, 6]:
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 PAGE {page_num} - TABLE SEPARATION ANALYSIS")
                logger.info(f"{'='*80}")
                
                # Show original vs cleaned content
                original_length = segment.metadata.get('original_text_length', len(segment.content))
                cleaned_length = segment.metadata.get('cleaned_text_length', len(segment.content))
                
                logger.info(f"\nContent Analysis:")
                logger.info(f"  - Original text length: {original_length} chars")
                logger.info(f"  - Cleaned text length: {cleaned_length} chars")
                logger.info(f"  - Reduction: {original_length - cleaned_length} chars "
                           f"({(1 - cleaned_length/original_length)*100:.1f}%)")
                
                # Show table boundaries
                boundaries = segment.metadata.get('table_boundaries', [])
                if boundaries:
                    logger.info(f"\nTable boundaries found: {len(boundaries)}")
                    for i, (start, end) in enumerate(boundaries):
                        logger.info(f"  - Table {i+1}: chars {start}-{end} "
                                   f"(length: {end-start})")
                
                # Show cleaned content
                logger.info(f"\n📝 CLEANED CONTENT (first 500 chars):")
                logger.info("-"*40)
                preview = segment.content[:500]
                logger.info(preview + "..." if len(segment.content) > 500 else preview)
                
                # Show extracted tables
                tables = segment.metadata.get('extracted_tables', [])
                if tables:
                    logger.info(f"\n📊 EXTRACTED TABLES: {len(tables)}")
                    for table in tables:
                        logger.info(f"\nTable {table['table_id']}:")
                        logger.info(f"  Type: {table.get('table_type', 'unknown')}")
                        logger.info(f"  Size: {table['row_count']}x{table['col_count']}")
                        logger.info(f"  Headers: {table['headers']}")
                        if table.get('data'):
                            logger.info(f"  First row: {table['data'][0]}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("📊 SEPARATION SUMMARY")
        logger.info(f"{'='*80}")
        
        pages_with_separation = 0
        total_reduction = 0
        
        for segment in segments:
            if segment.metadata.get('table_boundaries'):
                pages_with_separation += 1
                original = segment.metadata.get('original_text_length', 0)
                cleaned = segment.metadata.get('cleaned_text_length', 0)
                if original > 0:
                    total_reduction += (original - cleaned)
        
        logger.info(f"Pages with table separation: {pages_with_separation}")
        logger.info(f"Total characters removed: {total_reduction}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    document = test_table_separation()