#!/usr/bin/env python3
"""
Show complete RAW output and resulting segments
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor
from plugins.parsers.base_parser import Segment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_raw_to_segments():
    """Show RAW output and how it's converted to segments"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🔍 RAW OUTPUT → SEGMENTS DEMONSTRATION")
    logger.info("="*80)
    
    # Test pages 2 and 6
    test_pages = [2, 6]
    
    # Initialize SmolDocling for RAW output
    smol_config = {
        'max_pages': 6,
        'gpu_memory_utilization': 0.2
    }
    smol_client = VLLMSmolDoclingClient(smol_config)
    
    # Initialize advanced extractor
    layout_settings = {
        'use_layout': True,
        'table_x_tolerance': 3,
        'table_y_tolerance': 3,
        'text_x_tolerance': 5,
        'text_y_tolerance': 5
    }
    extractor = AdvancedPDFExtractor(layout_settings)
    
    # Load SmolDocling model
    logger.info("\n📦 Loading SmolDocling model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(smol_client.model_id)
    
    # Process with SmolDocling to get RAW output
    logger.info("\n🤖 Getting SmolDocling RAW output...")
    smol_result = smol_client.parse_pdf(pdf_path)
    
    for page_num in test_pages:
        logger.info(f"\n{'='*80}")
        logger.info(f"📄 PAGE {page_num} ANALYSIS")
        logger.info(f"{'='*80}")
        
        # Get SmolDocling result for this page
        smol_page = None
        for page in smol_result.pages:
            if page.page_number == page_num:
                smol_page = page
                break
        
        if smol_page:
            # Show RAW SmolDocling output
            logger.info(f"\n1️⃣ SMOLDOCLING RAW OUTPUT:")
            logger.info("="*40)
            
            # The raw V2T output should be in the page metadata
            raw_v2t = smol_page.layout_info.get('raw_v2t_output', '')
            if not raw_v2t and hasattr(smol_page, 'raw_content'):
                raw_v2t = smol_page.raw_content
            
            logger.info("--- START RAW V2T ---")
            logger.info(raw_v2t if raw_v2t else "[No raw output available]")
            logger.info("--- END RAW V2T ---")
            
            # Show SmolDocling parsing results
            logger.info(f"\n📊 SmolDocling Detection:")
            logger.info(f"  - Tables: {smol_page.table_count}")
            logger.info(f"  - Images: {smol_page.picture_count}")
            logger.info(f"  - Text blocks: {len(smol_page.text_blocks)}")
            logger.info(f"  - Complex layout: {smol_page.layout_info.get('complex_layout_detection', {}).get('is_complex_layout', False)}")
        
        # Get advanced extraction result
        logger.info(f"\n2️⃣ ADVANCED EXTRACTION (with BBox filtering):")
        logger.info("="*40)
        
        adv_result = extractor.extract_page_with_bbox(pdf_path, page_num)
        
        # Show extraction summary
        logger.info(f"\n📊 Extraction Summary:")
        logger.info(f"  - Tables found: {adv_result['metadata']['tables_found']}")
        logger.info(f"  - Table coverage: {adv_result['metadata']['table_coverage']}%")
        logger.info(f"  - Text length (filtered): {adv_result['metadata']['text_length']} chars")
        logger.info(f"  - Full text length: {adv_result['metadata']['full_text_length']} chars")
        
        # Create segments from the extraction
        logger.info(f"\n3️⃣ RESULTING SEGMENTS:")
        logger.info("="*40)
        
        segments = []
        segment_position = page_num * 1000
        
        # Create text segment (if there's meaningful text outside tables)
        if adv_result['text'].strip() and len(adv_result['text'].strip()) > 50:
            text_segment = Segment(
                content=adv_result['text'],
                segment_type='text',
                metadata={
                    'page_number': page_num,
                    'source_file': str(pdf_path),
                    'parser': 'advanced_extractor',
                    'content_type': 'text_outside_tables',
                    'char_count': len(adv_result['text']),
                    'extraction_method': 'bbox_filtered'
                },
                position=segment_position
            )
            segments.append(text_segment)
            
            logger.info(f"\n📝 TEXT SEGMENT:")
            logger.info(f"  - Type: {text_segment.segment_type}")
            logger.info(f"  - Length: {len(text_segment.content)} chars")
            logger.info(f"  - Content preview:")
            logger.info("-"*40)
            preview = text_segment.content[:300]
            logger.info(preview + "..." if len(text_segment.content) > 300 else preview)
        
        # Create table segments
        for i, table in enumerate(adv_result['tables']):
            segment_position += 10
            
            # Use layout-preserved text if available, otherwise format data
            if table.get('layout_text'):
                table_content = table['layout_text']
            else:
                # Format as markdown-style table
                lines = []
                # Headers
                lines.append(" | ".join(table['headers']))
                lines.append(" | ".join(["-"*10 for _ in table['headers']]))
                # Data
                for row in table['data']:
                    lines.append(" | ".join(str(cell) for cell in row))
                table_content = "\n".join(lines)
            
            table_segment = Segment(
                content=table_content,
                segment_type='table',
                metadata={
                    'page_number': page_num,
                    'source_file': str(pdf_path),
                    'parser': 'advanced_extractor',
                    'content_type': 'structured_table',
                    'table_id': table['table_id'],
                    'table_type': table['table_type'],
                    'bbox': table['bbox'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count'],
                    'structured_data': {
                        'headers': table['headers'],
                        'data': table['data']
                    }
                },
                position=segment_position
            )
            segments.append(table_segment)
            
            logger.info(f"\n📊 TABLE SEGMENT {table['table_id']}:")
            logger.info(f"  - Type: {table_segment.segment_type}")
            logger.info(f"  - Table type: {table['table_type']}")
            logger.info(f"  - Size: {table['row_count']}x{table['col_count']}")
            logger.info(f"  - BBox: {table['bbox']}")
            logger.info(f"  - Content preview:")
            logger.info("-"*40)
            preview = table_segment.content[:400]
            logger.info(preview + "..." if len(table_segment.content) > 400 else preview)
        
        # Summary for this page
        logger.info(f"\n📋 PAGE {page_num} SEGMENT SUMMARY:")
        logger.info(f"  - Total segments created: {len(segments)}")
        logger.info(f"  - Text segments: {sum(1 for s in segments if s.segment_type == 'text')}")
        logger.info(f"  - Table segments: {sum(1 for s in segments if s.segment_type == 'table')}")
        
        # Save segments as JSON for inspection
        segments_data = []
        for seg in segments:
            seg_data = {
                'segment_type': seg.segment_type,
                'position': seg.position,
                'content_length': len(seg.content),
                'content_preview': seg.content[:200] + "..." if len(seg.content) > 200 else seg.content,
                'metadata': seg.metadata
            }
            segments_data.append(seg_data)
        
        output_file = f"data/output/page_{page_num}_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n💾 Segments saved to: {output_file}")

if __name__ == "__main__":
    show_raw_to_segments()