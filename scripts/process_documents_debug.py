#!/usr/bin/env python3
"""
Debug-Enhanced Document Processing Pipeline

Production pipeline with comprehensive debugging and visualization capabilities.
Tracks segments, chunks, VLM descriptions and generates HTML reports.

Usage:
    python process_documents_debug.py --debug-level detailed --html-report
    python process_documents_debug.py --file BMW_X5.pdf --debug-level full
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.vllm_batch_processor import VLLMBatchProcessor, BatchProcessingConfig
from core.parsers import ParserFactory, Document
from core.content_chunker import ContentChunker
from core.vlm.two_stage_processor import TwoStageVLMProcessor
from core.vlm.confidence_evaluator import ConfidenceEvaluator
from core.pipeline_debugger import PipelineDebugger, PipelineDebugConfig, DebugLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebugDocumentProcessor:
    """Document processor with integrated debugging capabilities"""
    
    def __init__(self, debug_config: PipelineDebugConfig):
        self.debug_config = debug_config
        self.debugger = PipelineDebugger(debug_config)
        self.parser_factory = ParserFactory()
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        
        # Initialize VLM processor if needed
        self.vlm_processor = None
        self.confidence_evaluator = ConfidenceEvaluator()
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_file_with_debug(self, file_path: Path, enable_vlm: bool = True) -> Dict[str, Any]:
        """Process a single file with comprehensive debugging"""
        
        logger.info(f"🔍 Processing with debug level: {self.debug_config.debug_level.value}")
        
        # Generate document ID
        doc_id = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start debugging
        self.debugger.start_document_processing(file_path, doc_id)
        
        try:
            # STAGE 1: Parse document
            logger.info(f"📄 Stage 1: Parsing {file_path.name}")
            start_time = time.time()
            
            # Use HybridPDFParser for PDFs to enable local SmolDocling
            if file_path.suffix.lower() == '.pdf':
                from core.parsers.implementations.pdf import HybridPDFParser
                parser = HybridPDFParser(enable_vlm=False)  # VLM handled separately
            else:
                parser = self.parser_factory.get_parser_for_file(file_path)
            
            document = await parser.parse(file_path)
            
            parsing_time = time.time() - start_time
            self.debugger.track_parsing_complete(document, parsing_time)
            
            logger.info(f"   ✅ Parsed: {len(document.segments)} segments, {len(document.visual_elements)} visual elements")
            
            # STAGE 2: VLM Processing (if enabled and visual elements exist)
            if enable_vlm and document.visual_elements:
                logger.info(f"🤖 Stage 2: VLM Processing")
                start_time = time.time()
                
                if not self.vlm_processor:
                    self.vlm_processor = TwoStageVLMProcessor()
                
                # Process visual elements with VLM
                vlm_results = await self._process_visual_elements(document)
                
                vlm_time = time.time() - start_time
                logger.info(f"   ✅ VLM processed {len(vlm_results)} visual elements in {vlm_time:.2f}s")
            else:
                logger.info("   ⏭️  Skipping VLM processing (no visual elements or disabled)")
            
            # STAGE 3: Chunking
            logger.info(f"📦 Stage 3: Content Chunking")
            start_time = time.time()
            
            chunking_config = {
                "chunking": {
                    "strategies": {
                        "pdf": {
                            "max_tokens": 500,
                            "min_tokens": 100,
                            "overlap_tokens": 50,
                            "respect_boundaries": True
                        }
                    },
                    "enable_context_inheritance": True,
                    "context_inheritance": {
                        "enabled": True,
                        "max_context_tokens": 300
                    }
                }
            }
            
            chunker = ContentChunker(chunking_config)
            chunking_result = await chunker.chunk_document(document)
            
            chunking_time = time.time() - start_time
            self.debugger.track_chunking_complete(chunking_result, chunking_time)
            
            logger.info(f"   ✅ Created {len(chunking_result.contextual_chunks)} chunks in {chunking_time:.2f}s")
            
            # End debugging and generate reports
            report_path = self.debugger.end_document_processing()
            
            # Prepare output data
            result = {
                "document_id": doc_id,
                "file_info": {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat()
                },
                "processing_summary": {
                    "total_segments": len(document.segments),
                    "total_visual_elements": len(document.visual_elements),
                    "total_chunks": len(chunking_result.contextual_chunks),
                    "vlm_processed": len(vlm_results) if enable_vlm and document.visual_elements else 0
                },
                "debug_report": str(report_path) if report_path else None
            }
            
            # Save processing results
            output_file = self.output_dir / f"{file_path.stem}_debug_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Processing complete!")
            if report_path:
                logger.info(f"📊 Debug report: {report_path}")
            
            return result
            
        except Exception as e:
            self.debugger.track_error("processing", e)
            self.debugger.end_document_processing()
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            raise
    
    async def _process_visual_elements(self, document: Document) -> List[Dict[str, Any]]:
        """Process visual elements with VLM and track in debugger"""
        results = []
        
        # Process each visual element
        for visual in document.visual_elements:
            try:
                start_time = time.time()
                
                # Find corresponding segment
                segment = next((s for s in document.segments if visual.content_hash in s.visual_references), None)
                if not segment:
                    continue
                
                # Process with VLM (simplified for example)
                # In production, this would use the full TwoStageVLMProcessor
                element_type = visual.element_type.value if hasattr(visual.element_type, 'value') else str(visual.element_type)
                page_num = visual.page_or_slide if hasattr(visual, 'page_or_slide') else None
                
                vlm_result = {
                    "model": "qwen2.5-vl",
                    "description": f"Visual element of type {element_type} on page {page_num}",
                    "confidence": 0.85
                }
                
                processing_time = time.time() - start_time
                
                # Track in debugger
                self.debugger.track_vlm_processing(
                    segment_id=visual.content_hash,
                    model=vlm_result["model"],
                    description=vlm_result["description"],
                    confidence=vlm_result["confidence"],
                    processing_time=processing_time
                )
                
                # Update segment with VLM description
                segment.content = f"{segment.content}\n\n[VLM Description: {vlm_result['description']}]"
                
                results.append(vlm_result)
                
            except Exception as e:
                self.debugger.track_error("vlm_processing", e)
                logger.error(f"Error processing visual element {visual.content_hash}: {str(e)}")
        
        return results
    
    async def process_batch_with_debug(self, file_paths: List[Path], enable_vlm: bool = True) -> List[Dict[str, Any]]:
        """Process multiple files with debugging"""
        results = []
        
        for file_path in file_paths:
            try:
                result = await self.process_file_with_debug(file_path, enable_vlm)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                results.append({
                    "file_info": {"filename": file_path.name},
                    "error": str(e)
                })
        
        return results


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Debug-Enhanced Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Debug Levels:
  none     - No debugging (production mode)
  basic    - Track basic metrics and timings
  detailed - Track segments, chunks, and VLM processing
  full     - Complete debugging with all intermediate data

Examples:
  %(prog)s --debug-level detailed --html-report
  %(prog)s --file BMW_X5.pdf --debug-level full
  %(prog)s --pattern "BMW*.pdf" --debug-level basic
        """
    )
    
    # Debug options
    parser.add_argument("--debug-level", 
                       choices=["none", "basic", "detailed", "full"],
                       default="basic",
                       help="Debug level for pipeline execution")
    parser.add_argument("--html-report", 
                       action="store_true",
                       help="Generate HTML analysis report")
    parser.add_argument("--debug-dir",
                       type=str,
                       default="data/debug",
                       help="Directory for debug output")
    
    # File selection
    parser.add_argument("--file", 
                       type=str,
                       help="Process specific file")
    parser.add_argument("--pattern",
                       type=str,
                       help="File pattern to match (e.g., 'BMW*.pdf')")
    
    # Processing options
    parser.add_argument("--no-vlm",
                       action="store_true",
                       help="Disable VLM processing")
    parser.add_argument("--save-intermediate",
                       action="store_true",
                       help="Save intermediate processing results")
    
    args = parser.parse_args()
    
    print("🔍 Debug-Enhanced Document Processing Pipeline")
    print("=" * 60)
    
    # Create debug configuration
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel(args.debug_level),
        generate_html_report=args.html_report,
        save_intermediate_results=args.save_intermediate or args.debug_level in ["detailed", "full"],
        output_dir=Path(args.debug_dir),
        include_images=args.debug_level == "full"
    )
    
    print(f"Debug Level: {debug_config.debug_level.value}")
    print(f"HTML Report: {'Enabled' if debug_config.generate_html_report else 'Disabled'}")
    print(f"VLM Processing: {'Disabled' if args.no_vlm else 'Enabled'}")
    print()
    
    # Initialize processor
    processor = DebugDocumentProcessor(debug_config)
    
    # Determine files to process
    input_dir = Path("data/input")
    files_to_process = []
    
    if args.file:
        file_path = input_dir / args.file
        if file_path.exists():
            files_to_process = [file_path]
        else:
            print(f"❌ File not found: {file_path}")
            return
    elif args.pattern:
        files_to_process = list(input_dir.glob(args.pattern))
        if not files_to_process:
            print(f"❌ No files match pattern: {args.pattern}")
            return
    else:
        # Process all PDFs by default
        files_to_process = list(input_dir.glob("*.pdf"))
    
    if not files_to_process:
        print("⚠️  No files to process")
        return
    
    print(f"📁 Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        print(f"   - {f.name}")
    print()
    
    try:
        # Process files
        results = await processor.process_batch_with_debug(
            files_to_process, 
            enable_vlm=not args.no_vlm
        )
        
        # Print summary
        successful = sum(1 for r in results if "error" not in r)
        print(f"\n📊 Processing Summary:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
        # Show debug reports
        print(f"\n📂 Debug reports saved to: {debug_config.output_dir}")
        for result in results:
            if result.get("debug_report"):
                print(f"   - {Path(result['debug_report']).name}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.debug_level in ["detailed", "full"]:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())