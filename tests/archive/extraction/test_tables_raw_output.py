#!/usr/bin/env python3
"""
Test BMW PDF with detailed RAW output for tables
"""

import logging
from pathlib import Path
from datetime import datetime
from core.parsers.hybrid_pdf_parser import HybridPDFParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_tables_with_raw_output():
    """Process BMW PDF and show RAW output for pages with tables"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🚀 PROCESSING BMW PDF - FOKUS AUF TABELLEN MIT RAW OUTPUT")
    logger.info("="*80)
    
    # Configure parser - process all pages to find tables
    config = {
        'max_pages': 10,
        'gpu_memory_utilization': 0.2,
        'prefer_pdfplumber': True,
        'fallback_confidence_threshold': 0.8
    }
    
    parser = HybridPDFParser(config)
    
    # Load model
    logger.info("\n📦 Loading SmolDocling model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(parser.smoldocling_client.model_id)
    
    # Process document
    logger.info("\n📄 Processing BMW PDF with focus on RAW table output...")
    
    try:
        document = parser.parse(pdf_path)
        segments = document.segments
        
        logger.info(f"\n✅ Processing complete. Total segments: {len(segments)}")
        
        # Analyze each page for tables
        for segment in segments:
            page_num = segment.metadata.get('page_number', 0)
            has_tables = segment.metadata.get('has_tables', False)
            table_count = segment.metadata.get('table_count', 0)
            extracted_tables = segment.metadata.get('extracted_tables', [])
            
            # Show pages with tables or potential tables
            if has_tables or table_count > 0 or extracted_tables:
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 PAGE {page_num} - TABELLE(N) GEFUNDEN!")
                logger.info(f"{'='*80}")
                
                logger.info(f"Parser verwendet: {segment.metadata.get('parser_used', segment.metadata.get('parser', 'unknown'))}")
                logger.info(f"Anzahl Tabellen: {table_count}")
                
                # If pdfplumber was used, show extracted tables
                if extracted_tables:
                    logger.info("\n🔍 EXTRAHIERTE TABELLEN (pdfplumber):")
                    for i, table in enumerate(extracted_tables):
                        logger.info(f"\n--- TABELLE {i+1} ---")
                        logger.info(f"Typ: {table.get('table_type', 'unknown')}")
                        logger.info(f"Größe: {table['row_count']} Zeilen x {table['col_count']} Spalten")
                        logger.info(f"\nHEADERS:")
                        for j, header in enumerate(table['headers']):
                            logger.info(f"  [{j}]: {header}")
                        
                        logger.info(f"\nDATEN:")
                        for row_idx, row in enumerate(table['data']):
                            logger.info(f"  Zeile {row_idx + 1}: {row}")
                
                # Show raw content
                logger.info(f"\n📝 RAW CONTENT (erste 1500 Zeichen):")
                logger.info("-"*40)
                raw_preview = segment.content[:1500]
                logger.info(raw_preview + "..." if len(segment.content) > 1500 else raw_preview)
                
                # If SmolDocling detected tables, show that info
                if has_tables and table_count > 0:
                    logger.info(f"\n📊 SmolDocling erkannte {table_count} Tabelle(n)")
                    
            # Also check pages that might have complex layouts
            elif segment.metadata.get('complex_layout_detection', {}).get('is_complex_layout'):
                logger.info(f"\n{'='*80}")
                logger.info(f"⚠️  PAGE {page_num} - KOMPLEXES LAYOUT")
                logger.info(f"{'='*80}")
                logger.info(f"Confidence: {segment.metadata['complex_layout_detection']['confidence']}")
                logger.info(f"Parser: {segment.metadata.get('parser_used', 'unknown')}")
        
        logger.info("\n" + "="*80)
        logger.info("📊 ZUSAMMENFASSUNG")
        logger.info("="*80)
        
        # Count total tables
        total_tables = 0
        pages_with_tables = []
        
        for segment in segments:
            page_num = segment.metadata.get('page_number', 0)
            table_count = segment.metadata.get('table_count', 0)
            extracted_tables = segment.metadata.get('extracted_tables', [])
            
            if table_count > 0 or extracted_tables:
                total_tables += max(table_count, len(extracted_tables))
                pages_with_tables.append(page_num)
        
        logger.info(f"Gesamtzahl gefundener Tabellen: {total_tables}")
        logger.info(f"Seiten mit Tabellen: {pages_with_tables}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    document = test_tables_with_raw_output()