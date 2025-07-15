#!/usr/bin/env python3
"""
Test advanced hybrid parser with configurable pdfplumber modes
"""

import logging
from pathlib import Path
from datetime import datetime
from core.parsers.hybrid_pdf_parser import HybridPDFParser
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_advanced_modes():
    """Test different pdfplumber modes"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🧪 TESTING ADVANCED HYBRID PARSER MODES")
    logger.info("="*80)
    
    # Test configuration with mode 2 (always parallel)
    config = {
        'max_pages': 6,
        'gpu_memory_utilization': 0.2,
        'pdfplumber_mode': 2,  # Always run pdfplumber in parallel
        'separate_tables': True,
        'layout_settings': {
            'use_layout': True,
            'table_x_tolerance': 3,
            'table_y_tolerance': 3,
            'text_x_tolerance': 5,
            'text_y_tolerance': 5
        }
    }
    
    # Initialize advanced extractor
    extractor = AdvancedPDFExtractor(config['layout_settings'])
    
    # Test specific pages with advanced extraction
    test_pages = [2, 6]  # Pages with tables
    
    for page_num in test_pages:
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 TESTING PAGE {page_num} WITH ADVANCED EXTRACTION")
        logger.info(f"{'='*80}")
        
        # Extract with bbox filtering
        result = extractor.extract_page_with_bbox(pdf_path, page_num)
        
        # Show results
        logger.info(f"\n📊 Extraction Results:")
        logger.info(f"  - Tables found: {result['metadata']['tables_found']}")
        logger.info(f"  - Table coverage: {result['metadata']['table_coverage']}%")
        logger.info(f"  - Text outside tables: {result['metadata']['text_length']} chars")
        logger.info(f"  - Full text: {result['metadata']['full_text_length']} chars")
        logger.info(f"  - Text reduction: {result['metadata']['full_text_length'] - result['metadata']['text_length']} chars")
        
        # Show tables
        if result['tables']:
            logger.info(f"\n📊 EXTRACTED TABLES:")
            for table in result['tables']:
                logger.info(f"\nTable {table['table_id']}:")
                logger.info(f"  Type: {table['table_type']}")
                logger.info(f"  Size: {table['row_count']}x{table['col_count']}")
                logger.info(f"  BBox: {table['bbox']}")
                logger.info(f"  Headers: {table['headers']}")
                
                # Show layout-preserved text if available
                if table.get('layout_text'):
                    logger.info(f"\n  📝 LAYOUT-PRESERVED TABLE:")
                    logger.info("-"*40)
                    # Show first 500 chars
                    preview = table['layout_text'][:500]
                    logger.info(preview + "..." if len(table['layout_text']) > 500 else preview)
                else:
                    # Show data rows
                    logger.info(f"\n  📋 DATA ROWS:")
                    for i, row in enumerate(table['data'][:3]):  # First 3 rows
                        logger.info(f"    Row {i+1}: {row}")
                    if len(table['data']) > 3:
                        logger.info(f"    ... and {len(table['data']) - 3} more rows")
        
        # Show filtered text
        logger.info(f"\n📝 TEXT OUTSIDE TABLES (first 500 chars):")
        logger.info("-"*40)
        text_preview = result['text'][:500]
        logger.info(text_preview + "..." if len(result['text']) > 500 else result['text'])
    
    # Test mode comparison
    logger.info(f"\n{'='*80}")
    logger.info("🔄 TESTING DIFFERENT MODES")
    logger.info(f"{'='*80}")
    
    modes = {
        0: "Never use pdfplumber",
        1: "Use pdfplumber as fallback only",
        2: "Always run pdfplumber in parallel"
    }
    
    for mode, description in modes.items():
        logger.info(f"\n📋 Mode {mode}: {description}")
        
        # Create parser with specific mode
        mode_config = config.copy()
        mode_config['pdfplumber_mode'] = mode
        
        parser = HybridPDFParser(mode_config)
        
        # For demonstration, just log the mode
        # In real implementation, this would affect the parsing logic
        logger.info(f"   Parser initialized with mode {mode}")
        logger.info(f"   Would process pages accordingly...")

if __name__ == "__main__":
    test_advanced_modes()