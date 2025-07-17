#!/usr/bin/env python3
"""
Direct test of fallback extraction for BMW page 2
"""

import logging
from pathlib import Path
from core.parsers.fallback_extractors import PyPDF2TextExtractor, PDFPlumberExtractor
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_fallback_extraction():
    """Test fallback extraction on BMW page 2"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🧪 TESTING FALLBACK EXTRACTION FOR BMW PAGE 2")
    logger.info("="*80)
    
    # 1. First show what SmolDocling sees
    logger.info("\n1️⃣ SMOLDOCLING RESULT:")
    logger.info("-"*40)
    
    client = VLLMSmolDoclingClient(max_pages=1, gpu_memory_utilization=0.2)
    
    # Load model
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(client.model_id)
    
    # Process page 2 with SmolDocling
    import pdf2image
    pages = pdf2image.convert_from_path(pdf_path, first_page=2, last_page=2, dpi=300)
    if pages:
        page_result = client.process_pdf_page(pages[0], 2)
        logger.info(f"SmolDocling extracted: {len(page_result.text)} chars")
        logger.info(f"Complex layout detected: {page_result.layout_info.get('complex_layout_detection', {}).get('is_complex_layout', False)}")
    
    # 2. Test PyPDF2 extraction
    logger.info("\n2️⃣ PYPDF2 EXTRACTION:")
    logger.info("-"*40)
    
    pypdf2_extractor = PyPDF2TextExtractor()
    pypdf2_result = pypdf2_extractor.extract_page_text(pdf_path, 2)
    
    logger.info(f"Extracted text length: {len(pypdf2_result['text'])} chars")
    logger.info(f"Lines extracted: {pypdf2_result['metadata']['line_count']}")
    logger.info(f"Potential tables found: {len(pypdf2_result['potential_tables'])}")
    logger.info(f"Lists found: {len(pypdf2_result['lists'])}")
    
    # Check for expected content
    expected_keywords = ['Motorisierung', 'Weltpremiere', '320d', 'Highlights', 'Motor', 'Getriebe']
    found_keywords = [kw for kw in expected_keywords if kw in pypdf2_result['text']]
    logger.info(f"Found keywords: {found_keywords}")
    
    # Show sample of extracted text
    logger.info("\nExtracted text preview:")
    logger.info("-"*40)
    preview = pypdf2_result['text'][:500]
    logger.info(preview + "..." if len(pypdf2_result['text']) > 500 else preview)
    
    # Show detected tables
    if pypdf2_result['potential_tables']:
        logger.info(f"\nDetected {len(pypdf2_result['potential_tables'])} potential tables:")
        for i, table in enumerate(pypdf2_result['potential_tables'][:2]):  # Show first 2
            logger.info(f"\nTable {i+1} ({table['row_count']} rows):")
            for row in table['rows'][:3]:  # Show first 3 rows
                logger.info(f"  {row}")
    
    # 3. Try pdfplumber if available
    try:
        logger.info("\n3️⃣ PDFPLUMBER EXTRACTION (if available):")
        logger.info("-"*40)
        
        pdfplumber_extractor = PDFPlumberExtractor()
        pdfplumber_result = pdfplumber_extractor.extract_page_content(pdf_path, 2)
        
        if not pdfplumber_result.get('error'):
            logger.info(f"Extracted text length: {len(pdfplumber_result['text'])} chars")
            logger.info(f"Tables extracted: {len(pdfplumber_result['tables'])}")
            
            if pdfplumber_result['tables']:
                logger.info(f"\nExtracted {len(pdfplumber_result['tables'])} tables:")
                for i, table in enumerate(pdfplumber_result['tables'][:1]):  # Show first table
                    logger.info(f"\nTable {table['table_id']} ({table['row_count']}x{table['col_count']}):")
                    for row in table['rows'][:5]:  # Show first 5 rows
                        logger.info(f"  {' | '.join(str(cell) if cell else '' for cell in row)}")
    except Exception as e:
        logger.info(f"pdfplumber not available: {e}")
    
    # Save results
    with open('data/output/page2_pypdf2_extraction.txt', 'w', encoding='utf-8') as f:
        f.write(pypdf2_result['text'])
    logger.info(f"\n📝 PyPDF2 extraction saved to: data/output/page2_pypdf2_extraction.txt")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY:")
    logger.info("="*80)
    
    if len(found_keywords) >= 3:
        logger.info("✅ SUCCESS! PyPDF2 successfully extracted content from page 2!")
        logger.info("   The fallback approach works for complex layouts.")
    else:
        logger.info("⚠️  Partial success - some content extracted but may be incomplete")
    
    return pypdf2_result

if __name__ == "__main__":
    result = test_fallback_extraction()