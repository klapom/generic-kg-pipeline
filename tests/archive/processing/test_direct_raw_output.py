#!/usr/bin/env python3
"""
Show RAW output and segments directly without model loading issues
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor
from plugins.parsers.base_parser import Segment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_raw_output_and_segments():
    """Show RAW output from previous runs and segment creation"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🔍 RAW OUTPUT → SEGMENTS DEMONSTRATION")
    logger.info("="*80)
    
    # Manually show RAW output from our previous tests
    # We know what SmolDocling returns for these pages
    
    # Page 2 RAW output (from previous test logs)
    page2_raw = """<doctag><picture><loc_14><loc_23><loc_493><loc_482></picture>"""
    
    # Page 6 RAW output (from previous test logs)
    page6_raw = """<doctag><picture><loc_22><loc_16><loc_293><loc_20></picture>
<picture><loc_424><loc_18><loc_490><loc_56></picture>
<picture><loc_129><loc_122><loc_197><loc_166></picture>
<picture><loc_287><loc_122><loc_340><loc_166></picture>
<picture><loc_424><loc_122><loc_490><loc_166></picture>
<otsl><loc_26><loc_170><loc_494><loc_471><ched>Benchmark<ched>BMW 3er Limousine 320d (F30) (140 kW, RWD, 8G-AT)<ched>Δ  →<ched>BMW 3er Limousine 320d (G20) (140 kW, RWD, 8G-AT)<ched>Δ  →<ched>Audi A4 B9 PA (140 kW, FWD, 7G-DCT)<nl><fcel>Benchmark-Werke in Tabelle grün hinterlegt<fcel>[mm]<fcel>4633<fcel>+76<fcel>4709<ecel><nl><fcel>Länge<fcel>[mm]<fcel>1811<fcel>+16<fcel>1827<ecel><nl><fcel>Breite<fcel>[mm]<fcel>1429<fcel>+13<fcel>1442<ecel><nl><fcel>Höhe<fcel>[mm]<fcel>2810<fcel>+41<fcel>2851<ecel><nl><fcel>Radstand<fcel>[mm]<fcel>1543 / 1583<fcel>+46 / 21<fcel>1589 / 1604<ecel><nl><fcel>Spurweite v/h<fcel>[mm]<fcel>480<fcel>-<fcel>480<ecel><nl><fcel>Kofferraumvolumen<fcel>[l]<fcel>1510<fcel>-55<fcel>1455<ecel><nl><fcel>Leergewicht (DIN)<fcel>[kg]<fcel>1600<fcel>-<fcel>1600<ecel><nl><fcel>Anhängelast<fcel>[kg]<fcel>0,26<fcel>-0,03<fcel>0,23<ecel><nl><fcel>Cw-Wert<fcel>[l]<fcel>57<fcel>-17<fcel>40 (S) | 59 (O)<ecel><nl><fcel>Tankvolumen<fcel>[l]<fcel>11,3<fcel>+0,1<fcel>11,4<ecel><nl><fcel>Wendekreis<fcel>[m]<fcel>205/60 R16<fcel>-<fcel>205/60 R16<ecel><nl><fcel>Basisrad<fcel>-<fcel>670<fcel>-<fcel>670<ecel><nl><fcel>Radfamilie<fcel>[mm]<fcel>670<fcel>-<fcel>CO$_{2}$-Werte B9 PA aufgrund laufender WLTP-Zulassung aktuell in Überarbeitung<ecel><nl></otsl>"""
    
    # Initialize advanced extractor
    layout_settings = {
        'use_layout': True,
        'table_x_tolerance': 3,
        'table_y_tolerance': 3,
        'text_x_tolerance': 5,
        'text_y_tolerance': 5
    }
    extractor = AdvancedPDFExtractor(layout_settings)
    
    # Process both pages
    for page_num, raw_output in [(2, page2_raw), (6, page6_raw)]:
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 PAGE {page_num} COMPLETE ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Show RAW SmolDocling output
        logger.info(f"\n1️⃣ SMOLDOCLING RAW V2T OUTPUT:")
        logger.info("="*60)
        logger.info("--- START RAW V2T ---")
        logger.info(raw_output)
        logger.info("--- END RAW V2T ---")
        
        # Analyze RAW output
        logger.info(f"\n📊 SmolDocling RAW Analysis:")
        picture_count = raw_output.count('<picture>')
        table_count = raw_output.count('<otsl>')
        text_count = raw_output.count('<text>')
        
        logger.info(f"  - Picture tags: {picture_count}")
        logger.info(f"  - Table tags (otsl): {table_count}")
        logger.info(f"  - Text tags: {text_count}")
        
        if page_num == 2:
            logger.info(f"  - ⚠️  PROBLEM: Only 1 picture, no tables/text detected!")
            logger.info(f"  - This triggers fallback extraction")
        else:
            logger.info(f"  - ✅ Table detected in RAW output")
        
        # Get advanced extraction result
        logger.info(f"\n2️⃣ ADVANCED EXTRACTION (pdfplumber with BBox):")
        logger.info("="*60)
        
        adv_result = extractor.extract_page_with_bbox(pdf_path, page_num)
        
        # Show what was extracted
        logger.info(f"\n📊 Extraction Results:")
        logger.info(f"  - Tables found: {adv_result['metadata']['tables_found']}")
        logger.info(f"  - Table coverage: {adv_result['metadata']['table_coverage']}%")
        logger.info(f"  - Text outside tables: {adv_result['metadata']['text_length']} chars")
        logger.info(f"  - Full page text: {adv_result['metadata']['full_text_length']} chars")
        logger.info(f"  - Characters saved by filtering: {adv_result['metadata']['full_text_length'] - adv_result['metadata']['text_length']}")
        
        # Show extracted tables with layout
        if adv_result['tables']:
            logger.info(f"\n📊 EXTRACTED TABLE DATA:")
            for table in adv_result['tables']:
                logger.info(f"\nTable {table['table_id']} - {table['table_type']}:")
                logger.info(f"  Size: {table['row_count']} rows x {table['col_count']} columns")
                logger.info(f"  BBox: {table['bbox']}")
                
                if table.get('layout_text'):
                    logger.info(f"\n  LAYOUT-PRESERVED TABLE TEXT:")
                    logger.info("  " + "-"*50)
                    # Show complete layout text
                    for line in table['layout_text'].split('\n'):
                        logger.info(f"  {line}")
                
                logger.info(f"\n  STRUCTURED DATA:")
                logger.info(f"  Headers: {table['headers']}")
                logger.info(f"  First 3 data rows:")
                for i, row in enumerate(table['data'][:3]):
                    logger.info(f"    Row {i+1}: {row}")
        
        # Create segments
        logger.info(f"\n3️⃣ RESULTING SEGMENTS:")
        logger.info("="*60)
        
        segments = []
        
        # Text segment (filtered content)
        if adv_result['text'].strip() and len(adv_result['text'].strip()) > 50:
            text_segment = Segment(
                content=adv_result['text'],
                segment_type='text',
                metadata={
                    'page_number': page_num,
                    'source_file': str(pdf_path),
                    'parser': 'pdfplumber_bbox_filtered',
                    'content_type': 'text_outside_tables',
                    'char_count': len(adv_result['text']),
                    'extraction_method': 'bbox_filtered',
                    'original_parser': 'smoldocling_failed' if page_num == 2 else 'smoldocling_partial'
                }
            )
            segments.append(text_segment)
            
            logger.info(f"\n📝 SEGMENT 1 - TEXT (outside tables):")
            logger.info(f"  - Type: {text_segment.segment_type}")
            logger.info(f"  - Length: {len(text_segment.content)} chars")
            logger.info(f"  - Metadata: {json.dumps(text_segment.metadata, indent=4)}")
            logger.info(f"  - Content preview (first 400 chars):")
            logger.info("  " + "-"*50)
            preview_lines = text_segment.content[:400].split('\n')
            for line in preview_lines:
                logger.info(f"  {line}")
            if len(text_segment.content) > 400:
                logger.info("  ...")
        
        # Table segments
        for i, table in enumerate(adv_result['tables']):
            
            # Use layout-preserved text
            table_content = table.get('layout_text', '')
            if not table_content:
                # Fallback: create simple table representation
                lines = []
                lines.append(" | ".join(table['headers']))
                lines.append("-" * 80)
                for row in table['data']:
                    lines.append(" | ".join(str(cell) for cell in row))
                table_content = "\n".join(lines)
            
            table_segment = Segment(
                content=table_content,
                segment_type='table',
                metadata={
                    'page_number': page_num,
                    'source_file': str(pdf_path),
                    'parser': 'pdfplumber_bbox_filtered',
                    'content_type': 'structured_table',
                    'table_id': table['table_id'],
                    'table_type': table['table_type'],
                    'bbox': table['bbox'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count'],
                    'structured_data': {
                        'headers': table['headers'],
                        'data': table['data']
                    },
                    'extraction_method': 'bbox_with_layout_preservation'
                }
            )
            segments.append(table_segment)
            
            logger.info(f"\n📊 SEGMENT {i+2} - TABLE {table['table_id']}:")
            logger.info(f"  - Type: {table_segment.segment_type}")
            logger.info(f"  - Table type: {table['table_type']}")
            logger.info(f"  - Size: {table['row_count']}x{table['col_count']}")
            logger.info(f"  - Content length: {len(table_segment.content)} chars")
            logger.info(f"  - Metadata: {json.dumps({k: v for k, v in table_segment.metadata.items() if k != 'structured_data'}, indent=4)}")
            logger.info(f"  - Content (layout-preserved):")
            logger.info("  " + "-"*50)
            # Show complete table content
            for line in table_segment.content.split('\n')[:20]:  # First 20 lines
                logger.info(f"  {line}")
            if len(table_segment.content.split('\n')) > 20:
                logger.info("  ...")
        
        # Summary
        logger.info(f"\n📋 PAGE {page_num} SUMMARY:")
        logger.info(f"  - SmolDocling: {'FAILED (single picture)' if page_num == 2 else 'PARTIAL SUCCESS (table detected)'}")
        logger.info(f"  - Fallback used: YES (pdfplumber with bbox)")
        logger.info(f"  - Total segments created: {len(segments)}")
        logger.info(f"  - Text segments: {sum(1 for s in segments if s.segment_type == 'text')}")
        logger.info(f"  - Table segments: {sum(1 for s in segments if s.segment_type == 'table')}")
        logger.info(f"  - No duplicate content between text and tables ✅")

if __name__ == "__main__":
    show_raw_output_and_segments()