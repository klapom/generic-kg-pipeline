#!/usr/bin/env python3
"""
Test specific pages (2 and 6) for table content with RAW output
"""

import logging
from pathlib import Path
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.parsers.fallback_extractors import PDFPlumberExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_page_2_and_6():
    """Test pages 2 and 6 specifically"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🚀 TESTING PAGES 2 & 6 - RAW TABLE OUTPUT")
    logger.info("="*80)
    
    # Test with pdfplumber for better table extraction
    logger.info("\n📑 USING PDFPLUMBER FOR TABLE EXTRACTION")
    
    extractor = PDFPlumberExtractor()
    
    # Process page 2
    logger.info("\n" + "="*80)
    logger.info("📄 PAGE 2 - MOTORISIERUNG TABELLE")
    logger.info("="*80)
    
    try:
        page2_content = extractor.extract_page_content(pdf_path, 2)
        
        logger.info(f"\n📝 RAW TEXT (erste 500 Zeichen):")
        logger.info("-"*40)
        logger.info(page2_content['text'][:500] + "...")
        
        if page2_content['tables']:
            logger.info(f"\n📊 GEFUNDENE TABELLEN: {len(page2_content['tables'])}")
            
            for i, table in enumerate(page2_content['tables']):
                logger.info(f"\n--- TABELLE {i+1} ---")
                logger.info(f"Typ: {table.get('table_type', 'unknown')}")
                logger.info(f"Größe: {table['row_count']} Zeilen x {table['col_count']} Spalten")
                
                logger.info(f"\n🔤 HEADERS:")
                for j, header in enumerate(table['headers']):
                    logger.info(f"  [{j}]: '{header}'")
                
                logger.info(f"\n📊 DATEN (alle Zeilen):")
                for row_idx, row in enumerate(table['data']):
                    logger.info(f"  Zeile {row_idx + 1}: {row}")
        
    except Exception as e:
        logger.error(f"Error processing page 2: {e}")
    
    # Process page 6
    logger.info("\n" + "="*80)
    logger.info("📄 PAGE 6 - TECHNISCHE DATEN VERGLEICH")
    logger.info("="*80)
    
    try:
        page6_content = extractor.extract_page_content(pdf_path, 6)
        
        logger.info(f"\n📝 RAW TEXT (erste 500 Zeichen):")
        logger.info("-"*40)
        logger.info(page6_content['text'][:500] + "...")
        
        if page6_content['tables']:
            logger.info(f"\n📊 GEFUNDENE TABELLEN: {len(page6_content['tables'])}")
            
            for i, table in enumerate(page6_content['tables']):
                logger.info(f"\n--- TABELLE {i+1} ---")
                logger.info(f"Typ: {table.get('table_type', 'unknown')}")
                logger.info(f"Größe: {table['row_count']} Zeilen x {table['col_count']} Spalten")
                
                logger.info(f"\n🔤 HEADERS:")
                for j, header in enumerate(table['headers']):
                    logger.info(f"  [{j}]: '{header}'")
                
                logger.info(f"\n📊 DATEN (erste 10 Zeilen):")
                for row_idx, row in enumerate(table['data'][:10]):
                    logger.info(f"  Zeile {row_idx + 1}: {row}")
                
                if len(table['data']) > 10:
                    logger.info(f"  ... und {len(table['data']) - 10} weitere Zeilen")
        
    except Exception as e:
        logger.error(f"Error processing page 6: {e}")
    
    # Now test with SmolDocling to see RAW output
    logger.info("\n" + "="*80)
    logger.info("🤖 SMOLDOCLING RAW OUTPUT COMPARISON")
    logger.info("="*80)
    
    # Configure SmolDocling
    config = {
        'max_pages': 6,  # Process up to page 6
        'gpu_memory_utilization': 0.2
    }
    
    client = VLLMSmolDoclingClient(config)
    
    # Load model
    logger.info("\n📦 Loading SmolDocling model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(client.model_id)
    
    # Process with SmolDocling
    try:
        logger.info("\n🔧 Processing with SmolDocling...")
        result = client.parse_pdf(pdf_path)
        
        # Find page 2 and 6 in results
        for page in result:
            if page.page_number in [2, 6]:
                logger.info(f"\n📄 SMOLDOCLING OUTPUT - PAGE {page.page_number}")
                logger.info("="*40)
                logger.info(f"Tables found: {page.table_count}")
                logger.info(f"Text length: {len(page.text)}")
                logger.info(f"\nRAW V2T OUTPUT (siehe Log oben für Details)")
                
    except Exception as e:
        logger.error(f"Error with SmolDocling: {e}")

if __name__ == "__main__":
    test_page_2_and_6()