#!/usr/bin/env python3
"""
Process all PDFs in input directory and write results to log file
WITHOUT displaying any content (privacy protection)
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor
from plugins.parsers.base_parser import Segment
import sys

# Setup file logging
log_filename = f"data/output/pipeline_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: Path, extractor: AdvancedPDFExtractor) -> dict:
    """Process a single PDF and return results"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*80}")
    
    results = {
        'filename': pdf_path.name,
        'pages_processed': 0,
        'total_segments': 0,
        'text_segments': 0,
        'table_segments': 0,
        'total_characters': 0,
        'pages': []
    }
    
    try:
        # Get page count
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        logger.info(f"Total pages: {total_pages}")
        
        # Process each page
        for page_num in range(1, min(total_pages + 1, 16)):  # Limit to 15 pages max
            logger.info(f"\nProcessing page {page_num}...")
            
            try:
                # Extract with bbox filtering
                result = extractor.extract_page_with_bbox(pdf_path, page_num)
                
                page_data = {
                    'page_number': page_num,
                    'tables_found': result['metadata']['tables_found'],
                    'text_length': result['metadata']['text_length'],
                    'full_text_length': result['metadata']['full_text_length'],
                    'table_coverage': result['metadata']['table_coverage'],
                    'segments': []
                }
                
                # Log RAW extraction info (without content)
                logger.info(f"Page {page_num} extraction summary:")
                logger.info(f"  - Tables found: {result['metadata']['tables_found']}")
                logger.info(f"  - Text outside tables: {result['metadata']['text_length']} chars")
                logger.info(f"  - Full page text: {result['metadata']['full_text_length']} chars")
                logger.info(f"  - Table coverage: {result['metadata']['table_coverage']}%")
                
                # Create segments
                # Text segment
                if result['text'].strip() and len(result['text'].strip()) > 20:
                    text_segment = {
                        'type': 'text',
                        'char_count': len(result['text']),
                        'metadata': {
                            'page_number': page_num,
                            'content_type': 'text_outside_tables'
                        }
                    }
                    page_data['segments'].append(text_segment)
                    results['text_segments'] += 1
                    results['total_characters'] += len(result['text'])
                    
                    logger.info(f"  - Created TEXT segment: {len(result['text'])} chars")
                
                # Table segments
                for i, table in enumerate(result['tables']):
                    table_content_length = len(table.get('layout_text', ''))
                    if not table_content_length:
                        # Calculate from structured data
                        table_content_length = sum(len(str(cell)) for row in table['data'] for cell in row)
                    
                    table_segment = {
                        'type': 'table',
                        'char_count': table_content_length,
                        'metadata': {
                            'page_number': page_num,
                            'table_id': table['table_id'],
                            'table_type': table['table_type'],
                            'row_count': table['row_count'],
                            'col_count': table['col_count'],
                            'bbox': table['bbox']
                        }
                    }
                    page_data['segments'].append(table_segment)
                    results['table_segments'] += 1
                    results['total_characters'] += table_content_length
                    
                    logger.info(f"  - Created TABLE segment {table['table_id']}: {table['row_count']}x{table['col_count']}, {table_content_length} chars")
                
                results['pages'].append(page_data)
                results['total_segments'] += len(page_data['segments'])
                results['pages_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary for {pdf_path.name}:")
        logger.info(f"  - Pages processed: {results['pages_processed']}")
        logger.info(f"  - Total segments: {results['total_segments']}")
        logger.info(f"  - Text segments: {results['text_segments']}")
        logger.info(f"  - Table segments: {results['table_segments']}")
        logger.info(f"  - Total characters: {results['total_characters']}")
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {str(e)}")
        results['error'] = str(e)
    
    return results

def main():
    """Process all PDFs in input directory"""
    
    logger.info("="*100)
    logger.info("PDF PROCESSING PIPELINE - PRIVACY MODE")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*100)
    
    # Initialize extractor
    layout_settings = {
        'use_layout': True,
        'table_x_tolerance': 3,
        'table_y_tolerance': 3,
        'text_x_tolerance': 5,
        'text_y_tolerance': 5
    }
    extractor = AdvancedPDFExtractor(layout_settings)
    
    # Find all PDFs in input directory
    input_dir = Path("data/input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    logger.info(f"\nFound {len(pdf_files)} PDF files to process:")
    for pdf in pdf_files:
        logger.info(f"  - {pdf.name}")
    
    # Process each PDF
    all_results = []
    for pdf_path in pdf_files:
        result = process_pdf(pdf_path, extractor)
        all_results.append(result)
    
    # Write detailed results to JSON
    json_output = f"data/output/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*100}")
    logger.info("PIPELINE PROCESSING COMPLETE")
    logger.info(f"{'='*100}")
    logger.info(f"Processed {len(pdf_files)} PDF files")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"JSON results: {json_output}")
    
    # Final summary
    total_pages = sum(r['pages_processed'] for r in all_results)
    total_segments = sum(r['total_segments'] for r in all_results)
    total_chars = sum(r['total_characters'] for r in all_results)
    
    logger.info(f"\nOVERALL STATISTICS:")
    logger.info(f"  - Total pages processed: {total_pages}")
    logger.info(f"  - Total segments created: {total_segments}")
    logger.info(f"  - Total characters extracted: {total_chars}")
    
    print(f"\n✅ Processing complete. Check log file: {log_filename}")

if __name__ == "__main__":
    main()