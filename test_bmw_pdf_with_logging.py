#!/usr/bin/env python3
"""
Test SmolDocling PDF processing with BMW PDF and detailed logging
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import json

# Set up logging to both console and file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data/output/BMW_SmolDocling_test_{timestamp}.log"

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

def test_bmw_pdf_processing():
    """Test SmolDocling PDF processing with BMW PDF"""
    try:
        logger.info("="*80)
        logger.info("🚗 Starting BMW PDF SmolDocling Test")
        logger.info(f"📝 Log file: {log_filename}")
        logger.info("="*80)
        
        # Find BMW PDF file
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF not found: {pdf_path}")
            return False
            
        logger.info(f"📄 Processing: {pdf_path.name}")
        logger.info(f"📏 File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
            
        # Import and create SmolDocling client
        logger.info("🔄 Starting import of VLLMSmolDoclingClient...")
        try:
            from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
            logger.info("✅ Import successful")
        except Exception as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            raise
        
        logger.info("📦 Creating SmolDocling client...")
        try:
            client = VLLMSmolDoclingClient(
                max_pages=5,  # Process first 5 pages of BMW PDF
                gpu_memory_utilization=0.15  # Slightly more memory for complex document
            )
            logger.info("✅ Client created successfully")
        except Exception as e:
            logger.error(f"❌ Client creation failed: {e}", exc_info=True)
            raise
        
        logger.info("🔄 Processing PDF with SmolDocling...")
        result = client.parse_pdf(pdf_path)
        
        if result.success:
            logger.info(f"✅ PDF processing successful!")
            logger.info(f"📊 Pages processed: {len(result.pages)}")
            logger.info(f"⏱️ Processing time: {result.processing_time_seconds:.1f}s")
            logger.info(f"⚡ Average time per page: {result.processing_time_seconds/len(result.pages):.1f}s")
            
            # Create detailed output JSON
            output_data = {
                "metadata": {
                    "pdf_file": str(pdf_path),
                    "processing_timestamp": timestamp,
                    "processing_time_seconds": result.processing_time_seconds,
                    "model_version": result.model_version,
                    "total_pages_processed": len(result.pages),
                    "success": result.success
                },
                "pages": []
            }
            
            # Process each page
            total_text_chars = 0
            total_tables = 0
            total_images = 0
            total_formulas = 0
            
            for i, page in enumerate(result.pages, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"📄 PAGE {i} ANALYSIS:")
                logger.info(f"{'='*80}")
                
                # Log raw V2T output
                raw_content = page.layout_info.get('raw_content', '')
                logger.info(f"📝 RAW V2T OUTPUT ({len(raw_content)} chars):")
                logger.info("--- START RAW DOCTAGS ---")
                logger.info(raw_content if raw_content else 'No raw content')
                logger.info("--- END RAW DOCTAGS ---")
                
                # Also log the raw vLLM response if available
                if 'vllm_response' in page.layout_info:
                    logger.info(f"\n🔧 RAW VLLM RESPONSE:")
                    logger.info("--- START VLLM RESPONSE ---")
                    logger.info(page.layout_info['vllm_response'])
                    logger.info("--- END VLLM RESPONSE ---")
                
                # Log extracted text
                logger.info(f"\n📝 EXTRACTED TEXT ({len(page.text)} chars):")
                logger.info("--- START EXTRACTED TEXT ---")
                logger.info(page.text)
                logger.info("--- END EXTRACTED TEXT ---")
                
                # Log tables
                if page.tables:
                    logger.info(f"\n📊 TABLES ({len(page.tables)} found):")
                    for j, table in enumerate(page.tables, 1):
                        logger.info(f"--- TABLE {j} ---")
                        logger.info(json.dumps(table, indent=2, ensure_ascii=False))
                
                # Log images
                if page.images:
                    logger.info(f"\n🖼️ IMAGES ({len(page.images)} found):")
                    for j, img in enumerate(page.images, 1):
                        logger.info(f"--- IMAGE {j} ---")
                        logger.info(json.dumps(img, indent=2, ensure_ascii=False))
                
                # Log formulas
                if page.formulas:
                    logger.info(f"\n🧮 FORMULAS ({len(page.formulas)} found):")
                    for j, formula in enumerate(page.formulas, 1):
                        logger.info(f"--- FORMULA {j} ---")
                        logger.info(json.dumps(formula, indent=2, ensure_ascii=False))
                
                # Update totals
                total_text_chars += len(page.text)
                total_tables += len(page.tables)
                total_images += len(page.images)
                total_formulas += len(page.formulas)
                
                # Add to output data
                page_data = {
                    "page_number": page.page_number,
                    "text_length": len(page.text),
                    "text_preview": page.text[:200] + "..." if len(page.text) > 200 else page.text,
                    "tables_count": len(page.tables),
                    "images_count": len(page.images),
                    "formulas_count": len(page.formulas),
                    "confidence_score": page.confidence_score,
                    "doctags_elements": page.layout_info.get('doctags_elements', {})
                }
                output_data["pages"].append(page_data)
                
                logger.info(f"\n📈 Page {i} Summary:")
                logger.info(f"   - Text characters: {len(page.text)}")
                logger.info(f"   - Tables: {len(page.tables)}")
                logger.info(f"   - Images: {len(page.images)}")
                logger.info(f"   - Formulas: {len(page.formulas)}")
                logger.info(f"   - Confidence score: {page.confidence_score}")
            
            # Overall summary
            logger.info(f"\n{'='*80}")
            logger.info("📊 OVERALL SUMMARY:")
            logger.info(f"{'='*80}")
            logger.info(f"✅ Total pages processed: {len(result.pages)}")
            logger.info(f"📝 Total text extracted: {total_text_chars} characters")
            logger.info(f"📊 Total tables found: {total_tables}")
            logger.info(f"🖼️ Total images found: {total_images}")
            logger.info(f"🧮 Total formulas found: {total_formulas}")
            logger.info(f"⏱️ Total processing time: {result.processing_time_seconds:.1f}s")
            logger.info(f"⚡ Average per page: {result.processing_time_seconds/len(result.pages):.1f}s")
            
            # Save JSON output
            json_output_path = f"data/output/BMW_SmolDocling_results_{timestamp}.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Results saved to: {json_output_path}")
            
            # Test conversion to Document
            logger.info("\n🔄 Converting to Document format...")
            document = client.convert_to_document(result, pdf_path)
            logger.info(f"✅ Document created: {len(document.segments)} segments")
            
            # Log segments summary
            segment_types = {}
            for seg in document.segments:
                segment_types[seg.segment_type] = segment_types.get(seg.segment_type, 0) + 1
            
            logger.info("\n📑 Segment types breakdown:")
            for seg_type, count in segment_types.items():
                logger.info(f"   - {seg_type}: {count}")
            
            return True
        else:
            logger.error(f"❌ PDF processing failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        logger.info(f"\n📝 Complete log saved to: {log_filename}")

if __name__ == "__main__":
    logger.info("🚀 Starting BMW SmolDocling PDF test with detailed logging...")
    success = test_bmw_pdf_processing()
    if success:
        logger.info("🎉 Test completed successfully!")
    else:
        logger.error("💥 Test failed!")