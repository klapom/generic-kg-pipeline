#!/usr/bin/env python3
"""
Process complete BMW PDF and display full document structure
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

def process_complete_document():
    """Process complete BMW PDF and show document structure"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🚀 PROCESSING COMPLETE BMW PDF WITH HYBRID PARSER")
    logger.info("="*80)
    
    # Create parser
    config = {
        'max_pages': 10,  # Process 10 pages of BMW document
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
        # Create Document metadata
        metadata = DocumentMetadata(
            title="BMW 3er G20 Preview",
            author="Burkhard Fuchs, Marvin Gregor",
            created_date=datetime.now(),
            file_size=pdf_path.stat().st_size,
            page_count=10,  # Processing up to 10 pages
            language="de",
            document_type=DocumentType.PDF
        )
        
        # Process with hybrid parser
        document = parser.parse(pdf_path)
        segments = document.segments
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Display results for each page
        logger.info(f"\n✅ Processing complete in {processing_time:.1f}s")
        logger.info(f"Total segments: {len(segments)}")
        
        # Create structured output
        document_structure = {
            "metadata": {
                "title": metadata.title,
                "author": metadata.author,
                "file_size": metadata.file_size,
                "page_count": metadata.page_count,
                "processing_time_seconds": processing_time
            },
            "pages": []
        }
        
        # Analyze each segment/page
        for segment in segments:
            page_num = segment.metadata.get('page_number', 0)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 PAGE {page_num}")
            logger.info(f"{'='*80}")
            
            page_data = {
                "page_number": page_num,
                "parser_used": segment.metadata.get('parser_used', 'unknown'),
                "segment_type": segment.segment_type,
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
                    logger.info(f"  - Table {table['table_id']}: {table['table_type']}, "
                               f"{table['row_count']}x{table['col_count']}")
                    logger.info(f"    Headers: {table['headers']}")
                    if table['data']:
                        logger.info(f"    First row: {table['data'][0]}")
            
            if segment.metadata.get('detected_images'):
                logger.info(f"Images: {len(segment.metadata['detected_images'])}")
                for i, img in enumerate(segment.metadata['detected_images'][:3], 1):
                    logger.info(f"  - Image {i}: {img.get('content', 'No location data')}")
            
            if segment.metadata.get('complex_layout_detection'):
                detection = segment.metadata['complex_layout_detection']
                if detection.get('is_complex_layout'):
                    logger.info(f"⚠️  Complex layout detected! Confidence: {detection.get('confidence')}")
            
            # Add to structured output
            document_structure["pages"].append(page_data)
            
            # Show content preview
            logger.info(f"\nContent preview:")
            logger.info("-"*40)
            preview = segment.content[:300]
            logger.info(preview + "..." if len(segment.content) > 300 else preview)
        
        # Save complete structure
        output_file = f"data/output/BMW_complete_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document_structure, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"\n📝 Complete document structure saved to: {output_file}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("📊 PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        
        # Count by parser
        smoldocling_pages = [s for s in segments if s.metadata.get('parser_used') != 'fallback']
        fallback_pages = [s for s in segments if s.metadata.get('parser_used') == 'fallback']
        
        logger.info(f"Total pages processed: {len(segments)}")
        logger.info(f"SmolDocling pages: {len(smoldocling_pages)}")
        logger.info(f"Fallback pages: {len(fallback_pages)}")
        
        # Total content stats
        total_chars = sum(len(s.content) for s in segments)
        total_tables = sum(len(s.metadata.get('extracted_tables', [])) for s in segments)
        total_images = sum(len(s.metadata.get('detected_images', [])) for s in segments)
        
        logger.info(f"\nContent statistics:")
        logger.info(f"  Total text: {total_chars} characters")
        logger.info(f"  Total tables: {total_tables}")
        logger.info(f"  Total images: {total_images}")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    document = process_complete_document()