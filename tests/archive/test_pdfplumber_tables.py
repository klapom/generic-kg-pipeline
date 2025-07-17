#!/usr/bin/env python3
"""
Test PDFPlumber table extraction for BMW page 2
"""

import logging
from pathlib import Path
import json
from core.parsers.fallback_extractors import PDFPlumberExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_table_extraction():
    """Test table extraction with pdfplumber"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    extractor = PDFPlumberExtractor()
    
    logger.info("="*80)
    logger.info("🧪 TESTING PDFPLUMBER TABLE EXTRACTION")
    logger.info("="*80)
    
    # Extract page 2
    result = extractor.extract_page_content(pdf_path, 2)
    
    if result.get('error'):
        logger.error(f"Extraction failed: {result['error']}")
        return
    
    logger.info(f"\n📊 Extraction Results:")
    logger.info(f"Text length: {result['metadata']['char_count']} chars")
    logger.info(f"Tables found: {result['metadata']['table_count']}")
    
    # Show text preview
    text_preview = result['text'][:300]
    logger.info(f"\n📝 Text preview:")
    logger.info("-"*40)
    logger.info(text_preview + "..." if len(result['text']) > 300 else text_preview)
    
    # Analyze tables
    for table in result['tables']:
        logger.info(f"\n📊 Table {table['table_id']}:")
        logger.info(f"   Type: {table.get('table_type', 'unknown')}")
        logger.info(f"   Size: {table['row_count']} rows x {table['col_count']} columns")
        logger.info(f"   Headers: {table['headers']}")
        
        # Show first few rows
        logger.info(f"\n   Data rows (first 5):")
        for i, row in enumerate(table['rows'][:5]):
            logger.info(f"   Row {i+1}: {row}")
        
        if table['row_count'] > 5:
            logger.info(f"   ... and {table['row_count'] - 5} more rows")
    
    # Save structured result
    output_file = 'data/output/page2_structured_tables.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # Remove raw_data for cleaner output
        clean_result = {
            'text': result['text'],
            'tables': [{k: v for k, v in table.items() if k != 'raw_data'} 
                      for table in result['tables']],
            'metadata': result['metadata']
        }
        json.dump(clean_result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📝 Structured results saved to: {output_file}")
    
    # Check if we got the expected content
    expected_keywords = ['Motorisierung', '320d', 'Getriebe', 'Motor']
    found_keywords = [kw for kw in expected_keywords if kw in result['text']]
    
    if len(found_keywords) >= 3:
        logger.info("\n✅ SUCCESS! Table content properly extracted")
    else:
        logger.warning("\n⚠️  Some expected content might be missing")
    
    return result

if __name__ == "__main__":
    result = test_table_extraction()