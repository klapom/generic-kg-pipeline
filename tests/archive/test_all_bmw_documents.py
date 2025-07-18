#!/usr/bin/env python3
"""
Test all BMW documents for SmolDocling repetition issues
Generates detailed logs for each document
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from core.parsers.implementations.pdf.hybrid_pdf_parser import HybridPDFParser

async def test_document(pdf_path: Path, log_dir: Path):
    """Test a single document and log results"""
    
    # Create document-specific log file
    doc_name = pdf_path.stem
    log_file = log_dir / f"{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure document-specific logger
    doc_logger = logging.getLogger(f"test_{doc_name}")
    doc_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    doc_logger.handlers = []
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    doc_logger.addHandler(file_handler)
    
    # Add console handler for summary
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    doc_logger.addHandler(console_handler)
    
    # Set specific module loggers to use our logger
    logging.getLogger('core.clients.vllm_smoldocling_final').handlers = []
    logging.getLogger('core.clients.vllm_smoldocling_final').addHandler(file_handler)
    logging.getLogger('core.clients.vllm_smoldocling_final').setLevel(logging.DEBUG)
    
    doc_logger.info(f"\n{'='*60}")
    doc_logger.info(f"📄 Testing: {pdf_path.name}")
    doc_logger.info(f"   Log file: {log_file}")
    doc_logger.info(f"{'='*60}\n")
    
    try:
        # Initialize parser
        parser = HybridPDFParser(
            config={
                'environment': 'production',  # Enables docling/SmolDocling
                'max_pages': 10,  # Process up to 10 pages per document
                'gpu_memory_utilization': 0.3
            },
            enable_vlm=False  # Disable VLM for faster testing
        )
        
        # Parse document
        start_time = datetime.now()
        document = await parser.parse(pdf_path)
        parse_time = (datetime.now() - start_time).total_seconds()
        
        # Collect statistics
        stats = {
            "document": pdf_path.name,
            "parse_time_seconds": parse_time,
            "total_segments": document.total_segments,
            "total_visual_elements": document.total_visual_elements,
            "pages_processed": len(set(seg.metadata.get('page_number', 0) for seg in document.segments)),
            "total_text_length": sum(len(seg.content) for seg in document.segments if seg.content),
            "repetition_warnings": [],
            "parsing_errors": []
        }
        
        # Check log for repetition warnings
        with open(log_file, 'r') as f:
            log_content = f.read()
            
            # Look for repetition warnings
            if "Detected token repetition bug" in log_content:
                stats["repetition_warnings"].append("Token repetition detected")
                doc_logger.warning("⚠️  TOKEN REPETITION BUG DETECTED!")
            
            if "Suspicious repetition:" in log_content:
                import re
                suspicious_matches = re.findall(r"Suspicious repetition: Tag '(\w+)' contains '([^']+)' (\d+) times", log_content)
                for match in suspicious_matches:
                    warning = f"Tag '{match[0]}' repeated '{match[1][:20]}...' {match[2]} times"
                    stats["repetition_warnings"].append(warning)
                    doc_logger.warning(f"⚠️  {warning}")
            
            # Look for parsing errors
            if "Failed to process page" in log_content:
                error_matches = re.findall(r"Failed to process page (\d+): (.+)", log_content)
                for match in error_matches:
                    error = f"Page {match[0]}: {match[1]}"
                    stats["parsing_errors"].append(error)
                    doc_logger.error(f"❌ {error}")
        
        # Summary
        doc_logger.info("\n📊 Summary:")
        doc_logger.info(f"   Parse time: {stats['parse_time_seconds']:.2f}s")
        doc_logger.info(f"   Pages processed: {stats['pages_processed']}")
        doc_logger.info(f"   Total segments: {stats['total_segments']}")
        doc_logger.info(f"   Total text: {stats['total_text_length']} characters")
        doc_logger.info(f"   Visual elements: {stats['total_visual_elements']}")
        
        if stats["repetition_warnings"]:
            doc_logger.warning(f"\n⚠️  Repetition Issues Found: {len(stats['repetition_warnings'])}")
            for warning in stats["repetition_warnings"][:5]:  # Show first 5
                doc_logger.warning(f"   - {warning}")
        else:
            doc_logger.info("✅ No repetition issues detected")
        
        if stats["parsing_errors"]:
            doc_logger.error(f"\n❌ Parsing Errors: {len(stats['parsing_errors'])}")
            for error in stats["parsing_errors"][:5]:  # Show first 5
                doc_logger.error(f"   - {error}")
        
        return stats
        
    except Exception as e:
        doc_logger.error(f"\n❌ FAILED: {e}")
        import traceback
        doc_logger.error(traceback.format_exc())
        
        return {
            "document": pdf_path.name,
            "error": str(e),
            "repetition_warnings": [],
            "parsing_errors": [str(e)]
        }
    finally:
        # Clean up handlers
        doc_logger.handlers = []
        logging.getLogger('core.clients.vllm_smoldocling_final').handlers = []

async def main():
    """Test all BMW documents"""
    
    print("\n" + "="*80)
    print("🚗 BMW Documents Repetition Test Suite")
    print("="*80)
    
    # Create log directory
    log_dir = Path("logs/bmw_repetition_tests")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Log directory: {log_dir}")
    
    # Find all BMW documents
    bmw_documents = [
        Path("data/input/Preview_BMW_3er_G20.pdf"),
        Path("data/input/Preview_BMW_X5_G05.pdf"),
        Path("data/input/Preview_BMW_8er_G14_G15.pdf")
    ]
    
    # Filter existing documents
    available_docs = [doc for doc in bmw_documents if doc.exists()]
    
    if not available_docs:
        print("❌ No BMW documents found!")
        return
    
    print(f"\n📄 Found {len(available_docs)} BMW documents:")
    for doc in available_docs:
        print(f"   - {doc.name}")
    
    # Test each document
    all_stats = []
    
    for i, doc_path in enumerate(available_docs, 1):
        print(f"\n[{i}/{len(available_docs)}] Testing {doc_path.name}...")
        stats = await test_document(doc_path, log_dir)
        all_stats.append(stats)
        
        # Brief pause between documents
        if i < len(available_docs):
            await asyncio.sleep(2)
    
    # Generate summary report
    summary_file = log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    total_warnings = sum(len(s.get("repetition_warnings", [])) for s in all_stats)
    total_errors = sum(len(s.get("parsing_errors", [])) for s in all_stats)
    
    print(f"\nDocuments tested: {len(all_stats)}")
    print(f"Total repetition warnings: {total_warnings}")
    print(f"Total parsing errors: {total_errors}")
    
    # Show documents with issues
    docs_with_issues = [s for s in all_stats if s.get("repetition_warnings") or s.get("parsing_errors")]
    
    if docs_with_issues:
        print(f"\n⚠️  Documents with issues ({len(docs_with_issues)}):")
        for stats in docs_with_issues:
            print(f"\n   📄 {stats['document']}:")
            if stats.get("repetition_warnings"):
                print(f"      Repetition warnings: {len(stats['repetition_warnings'])}")
                for warning in stats["repetition_warnings"][:3]:
                    print(f"      - {warning}")
            if stats.get("parsing_errors"):
                print(f"      Parsing errors: {len(stats['parsing_errors'])}")
                for error in stats["parsing_errors"][:3]:
                    print(f"      - {error}")
    else:
        print("\n✅ All documents processed successfully without repetition issues!")
    
    print(f"\n📁 Detailed logs saved to: {log_dir}")
    print(f"📊 Summary report: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())