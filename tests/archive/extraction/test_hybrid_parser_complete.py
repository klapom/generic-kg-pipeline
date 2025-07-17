#!/usr/bin/env python3
"""
Test HybridPDFParser with complete BMW PDF extraction including visual elements
Shows all segments and visual elements even without VLM processing
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

# Set up logging to both console and file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data/output/BMW_HybridParser_test_{timestamp}.log"

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
os.makedirs("data/output", exist_ok=True)
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Set up root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


def serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        return {k: serialize_for_json(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        return obj


async def test_hybrid_parser_with_visual_elements():
    """Test HybridPDFParser with complete document extraction"""
    try:
        logger.info("="*80)
        logger.info("🚗 Starting BMW PDF HybridParser Test with Visual Elements")
        logger.info(f"📝 Log file: {log_filename}")
        logger.info("="*80)
        
        # Find BMW PDF file
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            return False
            
        logger.info(f"📄 Processing: {pdf_path.name}")
        logger.info(f"📏 File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Import and create HybridPDFParser
        logger.info("🔄 Importing HybridPDFParser...")
        try:
            from core.parsers.implementations.pdf import HybridPDFParser
            logger.info("✅ Import successful")
        except Exception as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            raise
        
        # Configure parser for comprehensive extraction
        config = {
            'max_pages': 10,  # Process first 10 pages
            'gpu_memory_utilization': 0.2,
            'extraction_mode': 'smart',  # Smart fallback mode
            'fallback_confidence_threshold': 0.8,
            'extract_images': True,
            'extract_tables': True,
            'extract_formulas': True,
            'separate_tables': True,
            'use_bbox_filtering': True,
            'layout_settings': {
                'use_layout': True,
                'table_x_tolerance': 3,
                'table_y_tolerance': 3,
                'text_x_tolerance': 5,
                'text_y_tolerance': 5
            }
        }
        
        logger.info("📦 Creating HybridPDFParser...")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        try:
            parser = HybridPDFParser(config=config, enable_vlm=True)
            logger.info("✅ Parser created successfully")
        except Exception as e:
            logger.error(f"❌ Parser creation failed: {e}", exc_info=True)
            raise
        
        # Parse document
        logger.info("🔄 Parsing PDF with HybridPDFParser...")
        start_time = datetime.now()
        
        try:
            document = await parser.parse(pdf_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ PDF parsing successful!")
            logger.info(f"⏱️ Processing time: {processing_time:.1f}s")
            logger.info(f"📊 Total segments: {len(document.segments)}")
            logger.info(f"🖼️ Total visual elements: {len(document.visual_elements)}")
            
        except Exception as e:
            logger.error(f"❌ Parsing failed: {e}", exc_info=True)
            raise
        
        # Create detailed output structure
        output_data = {
            "metadata": {
                "pdf_file": str(pdf_path),
                "processing_timestamp": timestamp,
                "processing_time_seconds": processing_time,
                "parser": "HybridPDFParser",
                "config": config,
                "document_metadata": serialize_for_json(document.metadata)
            },
            "summary": {
                "total_segments": len(document.segments),
                "total_visual_elements": len(document.visual_elements),
                "segments_by_type": {},
                "visual_elements_by_type": {},
                "pages_with_content": set()
            },
            "segments": [],
            "visual_elements": []
        }
        
        # Process segments
        logger.info("\n" + "="*80)
        logger.info("📄 PROCESSING SEGMENTS")
        logger.info("="*80)
        
        for idx, segment in enumerate(document.segments):
            # Count by type
            seg_type = segment.segment_type
            output_data["summary"]["segments_by_type"][seg_type] = \
                output_data["summary"]["segments_by_type"].get(seg_type, 0) + 1
            
            # Track pages
            if segment.page_number:
                output_data["summary"]["pages_with_content"].add(segment.page_number)
            
            # Log segment details
            logger.info(f"\n--- Segment {idx + 1}/{len(document.segments)} ---")
            logger.info(f"Type: {segment.segment_type}")
            logger.info(f"Page: {segment.page_number}")
            logger.info(f"Content length: {len(segment.content)} chars")
            
            if segment.metadata:
                logger.info(f"Metadata: {json.dumps(segment.metadata, indent=2)}")
            
            # Show content preview
            content_preview = segment.content[:200] + "..." if len(segment.content) > 200 else segment.content
            logger.info(f"Content preview: {content_preview}")
            
            # Add to output
            segment_data = {
                "index": idx,
                "type": segment.segment_type,
                "page_number": segment.page_number,
                "content_length": len(segment.content),
                "content": segment.content,
                "metadata": segment.metadata,
                "visual_references": segment.visual_references
            }
            output_data["segments"].append(segment_data)
        
        # Process visual elements
        logger.info("\n" + "="*80)
        logger.info("🖼️ PROCESSING VISUAL ELEMENTS")
        logger.info("="*80)
        
        if document.visual_elements:
            for idx, visual_elem in enumerate(document.visual_elements):
                # Count by type
                elem_type = visual_elem.element_type.value
                output_data["summary"]["visual_elements_by_type"][elem_type] = \
                    output_data["summary"]["visual_elements_by_type"].get(elem_type, 0) + 1
                
                # Log visual element details
                logger.info(f"\n--- Visual Element {idx + 1}/{len(document.visual_elements)} ---")
                logger.info(f"Type: {visual_elem.element_type.value}")
                logger.info(f"Page/Slide: {visual_elem.page_or_slide}")
                logger.info(f"Content hash: {visual_elem.content_hash[:16]}...")
                
                # VLM description (would be populated if VLM was active)
                if visual_elem.vlm_description:
                    logger.info(f"VLM Description: {visual_elem.vlm_description}")
                else:
                    logger.info("VLM Description: [Would be generated by VLM if active]")
                
                # Analysis metadata
                if visual_elem.analysis_metadata:
                    logger.info(f"Analysis metadata: {json.dumps(visual_elem.analysis_metadata, indent=2)}")
                
                # Extracted data (for formulas)
                if visual_elem.extracted_data:
                    logger.info(f"Extracted data: {json.dumps(visual_elem.extracted_data, indent=2)}")
                
                # Add to output
                visual_data = {
                    "index": idx,
                    "type": visual_elem.element_type.value,
                    "page_or_slide": visual_elem.page_or_slide,
                    "content_hash": visual_elem.content_hash,
                    "vlm_description": visual_elem.vlm_description or "[Pending VLM processing]",
                    "analysis_metadata": visual_elem.analysis_metadata,
                    "extracted_data": visual_elem.extracted_data,
                    "segment_reference": visual_elem.segment_reference
                }
                output_data["visual_elements"].append(visual_data)
        else:
            logger.info("No visual elements extracted (check SmolDocling output)")
        
        # Convert pages set to list for JSON serialization
        output_data["summary"]["pages_with_content"] = sorted(list(output_data["summary"]["pages_with_content"]))
        
        # Save complete output to JSON
        output_file = f"data/output/BMW_HybridParser_complete_{timestamp}.json"
        logger.info(f"\n💾 Saving complete output to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📊 EXTRACTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total segments: {len(document.segments)}")
        logger.info(f"- Text segments: {output_data['summary']['segments_by_type'].get('text', 0)}")
        logger.info(f"- Table segments: {output_data['summary']['segments_by_type'].get('table', 0)}")
        logger.info(f"- Mixed segments: {output_data['summary']['segments_by_type'].get('mixed', 0)}")
        logger.info(f"\nTotal visual elements: {len(document.visual_elements)}")
        logger.info(f"- Images: {output_data['summary']['visual_elements_by_type'].get('IMAGE', 0)}")
        logger.info(f"- Formulas: {output_data['summary']['visual_elements_by_type'].get('FORMULA', 0)}")
        logger.info(f"- Charts: {output_data['summary']['visual_elements_by_type'].get('CHART', 0)}")
        logger.info(f"\nPages with content: {output_data['summary']['pages_with_content']}")
        logger.info(f"\n✅ Test completed successfully!")
        logger.info(f"📄 Full log: {log_filename}")
        logger.info(f"💾 JSON output: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_hybrid_parser_with_visual_elements())
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1)