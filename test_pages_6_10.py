#!/usr/bin/env python3
"""
Process pages 6-10 of BMW PDF focusing on table extraction
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from core.parsers.hybrid_pdf_parser import HybridPDFParser
from plugins.parsers.base_parser import Document, DocumentMetadata, DocumentType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger('vllm').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)

def process_pages_6_to_10():
    """Process pages 6-10 of BMW PDF"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🚀 PROCESSING BMW PDF PAGES 6-10 WITH HYBRID PARSER")
    logger.info("="*80)
    
    # Configure to start from page 6
    config = {
        'max_pages': 10,  # Process up to page 10
        'gpu_memory_utilization': 0.2,
        'prefer_pdfplumber': True,
        'fallback_confidence_threshold': 0.8
    }
    
    parser = HybridPDFParser(config)
    
    # Load SmolDocling model
    logger.info("\n📦 Loading SmolDocling model...")
    from core.vllm.model_manager import VLLMModelManager
    model_manager = VLLMModelManager()
    model_manager.load_model(parser.smoldocling_client.model_id)
    
    # Process document
    logger.info("\n📄 Processing BMW PDF...")
    start_time = datetime.now()
    
    try:
        # Process with hybrid parser
        document = parser.parse(pdf_path)
        segments = document.segments
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Display results for pages 6-10 only
        logger.info(f"\n✅ Processing complete in {processing_time:.1f}s")
        logger.info(f"Total segments: {len(segments)}")
        
        # Create structured output
        pages_6_10_structure = {
            "processing_info": {
                "pdf_file": str(pdf_path),
                "processing_time_seconds": processing_time,
                "total_pages_processed": len(segments)
            },
            "pages": []
        }
        
        # Analyze pages 6-10
        for segment in segments:
            page_num = segment.metadata.get('page_number', 0)
            
            # Only process pages 6-10
            if page_num < 6:
                continue
                
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 PAGE {page_num}")
            logger.info(f"{'='*80}")
            
            page_data = {
                "page_number": page_num,
                "parser_used": segment.metadata.get('parser_used', 'smoldocling'),
                "content_length": len(segment.content),
                "content_preview": segment.content[:200] + "..." if len(segment.content) > 200 else segment.content,
                "metadata": segment.metadata
            }
            
            # Display page info
            logger.info(f"Parser: {page_data['parser_used']}")
            logger.info(f"Content length: {page_data['content_length']} chars")
            
            # Check for special content
            if segment.metadata.get('extracted_tables'):
                logger.info(f"Tables: {len(segment.metadata['extracted_tables'])}")
                for table in segment.metadata['extracted_tables']:
                    logger.info(f"  - Table {table['table_id']}: {table.get('table_type', 'unknown')}, "
                               f"{table['row_count']}x{table['col_count']}")
                    logger.info(f"    Headers: {table['headers']}")
                    if table['data']:
                        logger.info(f"    First row: {table['data'][0]}")
                        if len(table['data']) > 1:
                            logger.info(f"    Second row: {table['data'][1]}")
            
            if segment.metadata.get('detected_images'):
                logger.info(f"Images: {len(segment.metadata['detected_images'])}")
            
            if segment.metadata.get('complex_layout_detection'):
                detection = segment.metadata['complex_layout_detection']
                if detection.get('is_complex_layout'):
                    logger.info(f"⚠️  Complex layout detected! Confidence: {detection.get('confidence')}")
            
            # Add to structured output
            pages_6_10_structure["pages"].append(page_data)
            
            # Show content preview
            logger.info(f"\nContent preview:")
            logger.info("-"*40)
            preview = segment.content[:300]
            logger.info(preview + "..." if len(segment.content) > 300 else preview)
        
        # Save pages 6-10 structure
        output_file = f"data/output/BMW_pages_6_10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pages_6_10_structure, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n📝 Pages 6-10 structure saved to: {output_file}")
        
        # Summary for pages 6-10
        logger.info(f"\n{'='*80}")
        logger.info("📊 PAGES 6-10 SUMMARY")
        logger.info(f"{'='*80}")
        
        # Count tables and complex layouts
        table_count = 0
        complex_pages = []
        
        for page in pages_6_10_structure["pages"]:
            if page['metadata'].get('extracted_tables'):
                table_count += len(page['metadata']['extracted_tables'])
            if page['metadata'].get('complex_layout_detection', {}).get('is_complex_layout'):
                complex_pages.append(page['page_number'])
        
        logger.info(f"Pages analyzed: {len(pages_6_10_structure['pages'])}")
        logger.info(f"Total tables found: {table_count}")
        logger.info(f"Complex layout pages: {complex_pages}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    document = process_pages_6_to_10()