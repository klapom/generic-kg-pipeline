#!/usr/bin/env python3
"""
Enhanced Debug Document Processing Pipeline

Production pipeline with comprehensive debugging showing:
- Full SmolDocling content extraction
- Visual elements with actual images
- VLM comparisons (Qwen2.5-VL, LLaVA, Pixtral)
- Contextual chunks

Based on successful test cases from VLM comparison tests.
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

from core.parsers import ParserFactory, Document
from core.parsers.interfaces.data_models import VisualElement
from core.parsers.implementations.pdf import HybridPDFParser
from core.content_chunker import ContentChunker
from core.pipeline_debugger_enhanced import (
    EnhancedPipelineDebugger, PipelineDebugConfig, DebugLevel
)

# Import VLM clients from successful tests
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient as VLLMSmolDoclingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedDebugDocumentProcessor:
    """Document processor with enhanced debugging and real VLM integration"""
    
    def __init__(self, debug_config: PipelineDebugConfig, vlm_models: List[str] = None):
        self.debug_config = debug_config
        self.debugger = EnhancedPipelineDebugger(debug_config)
        self.parser_factory = ParserFactory()
        self.vlm_models = vlm_models or ["mock"]
        
        # Initialize VLM clients
        self.vlm_clients = None
        self.init_vlm_clients()
        
        # Directories
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def init_vlm_clients(self):
        """Initialize VLM clients for multi-model comparison"""
        logger.info(f"Initializing VLM clients: {self.vlm_models}")
        
        if "mock" in self.vlm_models:
            logger.info("📋 Using mock VLM processing for faster execution")
            self.vlm_clients = None
            return
        
        try:
            self.vlm_clients = {}
            
            if "qwen" in self.vlm_models:
                logger.info("Loading Qwen2.5-VL...")
                self.vlm_clients['qwen2.5-vl'] = TransformersQwen25VLClient(
                    temperature=0.2,
                    max_new_tokens=512
                )
                
            if "llava" in self.vlm_models:
                logger.info("Loading LLaVA...")
                self.vlm_clients['llava'] = TransformersLLaVAClient(
                    model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                    load_in_8bit=True,
                    temperature=0.2,
                    max_new_tokens=512
                )
                
            if "pixtral" in self.vlm_models:
                logger.info("Loading Pixtral...")
                self.vlm_clients['pixtral'] = TransformersPixtralClient(
                    temperature=0.2,
                    max_new_tokens=512,
                    load_in_8bit=True
                )
            
            logger.info(f"✅ {len(self.vlm_clients)} VLM clients initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize VLM clients: {e}")
            logger.warning("Will use mock VLM processing")
            self.vlm_clients = None
    
    async def process_file_with_enhanced_debug(self, file_path: Path, timeout: int = 900) -> Dict[str, Any]:
        """Process a single file with enhanced debugging
        
        Args:
            file_path: Path to the file to process
            timeout: Maximum processing time in seconds (default: 900 = 15 minutes)
        """
        
        logger.info(f"🔍 Processing with enhanced debug level: {self.debug_config.debug_level.value}")
        logger.info(f"⏱️  Timeout: {timeout}s per file")
        
        # Generate document ID
        doc_id = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start debugging
        self.debugger.start_document_processing(file_path, doc_id)
        
        # Track start time for timeout
        overall_start_time = time.time()
        
        try:
            # STAGE 1: Parse document with SmolDocling
            logger.info(f"📄 Stage 1: Parsing {file_path.name} with SmolDocling")
            start_time = time.time()
            
            # Use HybridPDFParser for PDFs to get SmolDocling content
            smoldocling_result = None
            if file_path.suffix.lower() == '.pdf':
                # HybridPDFParser already uses SmolDocling internally
                parser = HybridPDFParser(enable_vlm=False)
                logger.info("   📑 Using HybridPDFParser with local vLLM SmolDocling")
            else:
                parser = self.parser_factory.get_parser_for_file(file_path)
            
            document = await parser.parse(file_path)
            
            parsing_time = time.time() - start_time
            self.debugger.track_parsing_complete(document, parsing_time, smoldocling_result)
            
            logger.info(f"   ✅ Parsed: {len(document.segments)} segments, {len(document.visual_elements)} visual elements")
            
            # Log sample content to verify extraction
            if document.segments:
                sample_content = document.segments[0].content[:200] if document.segments[0].content else "No content"
                logger.info(f"   📝 Sample content: {sample_content}...")
            
            # Check timeout
            elapsed = time.time() - overall_start_time
            if elapsed > timeout:
                logger.warning(f"⏱️ Timeout approaching after parsing ({elapsed:.1f}s), skipping VLM processing")
                # Still generate report with what we have
            else:
                # STAGE 2: VLM Processing for visual elements
                if document.visual_elements and self.vlm_clients:
                    logger.info(f"🤖 Stage 2: VLM Processing for {len(document.visual_elements)} visual elements")
                    start_time = time.time()
                    
                    await self._process_visual_elements_multi_vlm(document)
                    
                    vlm_time = time.time() - start_time
                    logger.info(f"   ✅ VLM processing completed in {vlm_time:.2f}s")
                else:
                    if not document.visual_elements:
                        logger.info("   ⏭️  No visual elements to process")
                    else:
                        logger.info("   ⏭️  VLM clients not available, using mock processing")
                        await self._process_visual_elements_mock(document)
            
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
                    "context": {
                        "inherit_metadata": True,  # This is what ContentChunker looks for
                        "enabled": True,
                        "max_context_tokens": 300,
                        "use_llm_for_context": True,
                        "llm_instruction": "Extract key concepts and relationships from this text for context inheritance"
                    },
                    "context_inheritance": {
                        "enabled": True,
                        "max_context_tokens": 300,
                        "use_llm_for_context": True,
                        "llm_instruction": "Extract key concepts and relationships from this text for context inheritance"
                    }
                }
            }
            
            chunker = ContentChunker(chunking_config)
            # Provide task template to enable context inheritance
            task_template = "Process this document for knowledge extraction and understanding"
            chunking_result = await chunker.chunk_document(document, task_template)
            
            chunking_time = time.time() - start_time
            self.debugger.track_chunking_complete(chunking_result, chunking_time)
            
            logger.info(f"   ✅ Created {len(chunking_result.contextual_chunks)} chunks in {chunking_time:.2f}s")
            
            # Generate enhanced HTML report
            report_path = self.debugger.generate_enhanced_html_report()
            
            # Also save standard debug data
            self.debugger.end_document_processing()
            
            # Prepare result
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
                    "vlm_models_used": list(self.vlm_clients.keys()) if self.vlm_clients else []
                },
                "enhanced_report": str(report_path) if report_path else None
            }
            
            # Save processing results
            output_file = self.output_dir / f"{file_path.stem}_enhanced_debug_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Processing complete!")
            if report_path:
                logger.info(f"📊 Enhanced report: {report_path}")
            
            return result
            
        except Exception as e:
            self.debugger.track_error("processing", e)
            logger.error(f"❌ Error processing {file_path.name}: {str(e)}")
            
            # Still try to generate report with what we have
            try:
                report_path = self.debugger.generate_enhanced_html_report()
                logger.info(f"📊 Partial report generated despite error: {report_path}")
            except Exception as report_error:
                logger.error(f"❌ Failed to generate report: {report_error}")
            
            self.debugger.end_document_processing()
            
            # Return error result
            return {
                "document_id": doc_id,
                "file_info": {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat()
                },
                "error": str(e),
                "processing_time": time.time() - overall_start_time
            }
    
    async def _process_visual_elements_multi_vlm(self, document: Document) -> None:
        """Process visual elements with Qwen2.5 and Pixtral (fallback)"""
        
        # Memory management
        import torch
        import gc
        
        processed_count = 0
        batch_size = 8  # Process in batches to manage memory
        
        for i, visual in enumerate(document.visual_elements):
            try:
                # Find corresponding segment
                segment = next((s for s in document.segments if visual.content_hash in s.visual_references), None)
                if not segment:
                    continue
                
                # Process with each VLM
                results = {}
                
                # Try Qwen2.5 first
                if 'qwen2.5-vl' in self.vlm_clients:
                    try:
                        start_time = time.time()
                        logger.info(f"   🎯 Processing visual element {i+1}/{len(document.visual_elements)} with Qwen2.5-VL...")
                        logger.debug(f"      Visual hash: {visual.content_hash}")
                        logger.debug(f"      Element type: {visual.element_type}")
                        logger.debug(f"      Page/slide: {getattr(visual, 'page_or_slide', 'unknown')}")
                        
                        # Analyze visual
                        analysis_result = await self.vlm_clients['qwen2.5-vl'].analyze_visual(
                            image_data=visual.raw_data,
                            element_type=visual.element_type,
                            analysis_focus="comprehensive"
                        )
                        
                        processing_time = time.time() - start_time
                        
                        results['qwen2.5-vl'] = {
                            'success': True,
                            'description': analysis_result.description,
                            'confidence': analysis_result.confidence,
                            'ocr_text': analysis_result.ocr_text,
                            'extracted_data': analysis_result.extracted_data,
                            'processing_time': processing_time
                        }
                        
                        logger.info(f"   ✅ Qwen2.5-VL: {analysis_result.confidence:.2%} confidence")
                        
                        # If Qwen2.5 has low confidence, also try Pixtral
                        if analysis_result.confidence < 0.7 and 'pixtral' in self.vlm_clients:
                            logger.info(f"   🔄 Low confidence ({analysis_result.confidence:.2f}), trying Pixtral as fallback...")
                            logger.debug(f"      Qwen2.5 description preview: {results['qwen2.5-vl']['description'][:100]}...")
                            await self._try_pixtral_fallback(visual, results)
                            
                    except Exception as e:
                        logger.error(f"   ❌ Qwen2.5-VL failed: {str(e)}")
                        results['qwen2.5-vl'] = {
                            'success': False,
                            'error': str(e),
                            'processing_time': time.time() - start_time
                        }
                        
                        # Fallback to Pixtral
                        if 'pixtral' in self.vlm_clients:
                            logger.info(f"   🔄 Falling back to Pixtral...")
                            await self._try_pixtral_fallback(visual, results)
                
                # If no Qwen, try Pixtral directly
                elif 'pixtral' in self.vlm_clients:
                    await self._try_pixtral_fallback(visual, results)
                
                # Track results in debugger
                self.debugger.track_vlm_processing_multi(visual.content_hash, results)
                
                # Update segment with best description
                best_result = max(
                    [(k, v) for k, v in results.items() if v.get('success', False)],
                    key=lambda x: x[1].get('confidence', 0),
                    default=(None, None)
                )
                
                if best_result[1]:
                    segment.content += f"\n\n[VLM Analysis ({best_result[0]})]:\n{best_result[1]['description']}"
                
                # Memory management after each batch
                processed_count += 1
                if processed_count % batch_size == 0:
                    # Check GPU memory usage
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                        memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                        logger.info(f"   💾 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                        
                        # Clean up if memory usage is high
                        if memory_allocated > 14.0:  # 14GB threshold
                            logger.warning(f"   ⚠️ High GPU memory usage detected, triggering cleanup...")
                            torch.cuda.empty_cache()
                            gc.collect()
                            await asyncio.sleep(1)  # Give GPU time to release memory
                            logger.info(f"   ✅ Memory cleanup completed")
                
            except Exception as e:
                logger.error(f"Error processing visual element: {str(e)}")
    
    async def _try_pixtral_fallback(self, visual: VisualElement, results: Dict[str, Any]) -> None:
        """Try Pixtral as fallback VLM"""
        try:
            start_time = time.time()
            
            analysis_result = await self.vlm_clients['pixtral'].analyze_visual(
                image_data=visual.raw_data,
                element_type=visual.element_type,
                analysis_focus="comprehensive"
            )
            
            processing_time = time.time() - start_time
            
            results['pixtral'] = {
                'success': True,
                'description': analysis_result.description,
                'confidence': analysis_result.confidence,
                'ocr_text': analysis_result.ocr_text,
                'extracted_data': analysis_result.extracted_data,
                'processing_time': processing_time
            }
            
            logger.info(f"   ✅ Pixtral: {analysis_result.confidence:.2%} confidence")
            
        except Exception as e:
            logger.error(f"   ❌ Pixtral failed: {str(e)}")
            results['pixtral'] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def _process_visual_elements_mock(self, document: Document) -> None:
        """Mock VLM processing when real models not available"""
        
        for visual in document.visual_elements:
            try:
                segment = next((s for s in document.segments if visual.content_hash in s.visual_references), None)
                if not segment:
                    continue
                
                # Create mock results for demonstration
                element_type = visual.element_type.value if hasattr(visual.element_type, 'value') else str(visual.element_type)
                page_num = visual.page_or_slide if hasattr(visual, 'page_or_slide') else None
                
                mock_results = {
                    'qwen2.5-vl': {
                        'success': True,
                        'description': f"[Mock] This is a {element_type} on page {page_num}. Qwen2.5-VL would provide detailed OCR and object detection here.",
                        'confidence': 0.92,
                        'ocr_text': "[Mock OCR text would appear here]",
                        'processing_time': 1.2
                    },
                    'llava': {
                        'success': True,
                        'description': f"[Mock] LLaVA analysis of {element_type}: Comprehensive scene understanding and detailed description would appear here.",
                        'confidence': 0.88,
                        'processing_time': 1.5
                    },
                    'pixtral': {
                        'success': True,
                        'description': f"[Mock] Pixtral analysis: Basic visual understanding of the {element_type}.",
                        'confidence': 0.75,
                        'processing_time': 0.8
                    }
                }
                
                self.debugger.track_vlm_processing_multi(visual.content_hash, mock_results)
                
                # Update segment
                segment.content += f"\n\n[Mock VLM Analysis]: Visual element of type {element_type} on page {page_num}"
                
            except Exception as e:
                logger.error(f"Error in mock VLM processing: {str(e)}")
    
    async def process_batch_with_enhanced_debug(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files with enhanced debugging"""
        results = []
        
        for file_path in file_paths:
            try:
                result = await self.process_file_with_enhanced_debug(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                results.append({
                    "file_info": {"filename": file_path.name},
                    "error": str(e)
                })
        
        return results
    
    def cleanup(self):
        """Clean up VLM clients"""
        if self.vlm_clients:
            logger.info("Cleaning up VLM clients...")
            for client in self.vlm_clients.values():
                try:
                    client.cleanup()
                except:
                    pass


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Debug Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This enhanced pipeline shows:
- Full SmolDocling content extraction
- Visual elements with actual images
- VLM comparisons (Qwen2.5-VL, LLaVA, Pixtral)
- Contextual chunks with inheritance

Debug Levels:
  basic    - Track basic metrics and timings
  detailed - Full content and VLM comparisons
  full     - Everything including page images

Examples:
  %(prog)s --file BMW_X5.pdf --debug-level detailed
  %(prog)s --pattern "BMW*.pdf" --debug-level full
        """
    )
    
    # Debug options
    parser.add_argument("--debug-level", 
                       choices=["basic", "detailed", "full"],
                       default="detailed",
                       help="Debug level for pipeline execution")
    parser.add_argument("--debug-dir",
                       type=str,
                       default="data/debug/enhanced",
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
                       help="Skip VLM processing")
    parser.add_argument("--vlm-models",
                       nargs="+",
                       choices=["qwen", "llava", "pixtral", "mock"],
                       default=["mock"],
                       help="VLM models to use (default: mock for faster processing)")
    
    args = parser.parse_args()
    
    print("🔍 Enhanced Debug Document Processing Pipeline")
    print("=" * 60)
    
    # Create debug configuration
    debug_config = PipelineDebugConfig(
        debug_level=DebugLevel(args.debug_level),
        generate_html_report=True,
        track_segments=True,
        track_chunks=True,
        track_vlm_descriptions=True,
        save_intermediate_results=args.debug_level in ["detailed", "full"],
        output_dir=Path(args.debug_dir),
        include_images=args.debug_level == "full"
    )
    
    print(f"Debug Level: {debug_config.debug_level.value}")
    print(f"Output Directory: {debug_config.output_dir}")
    
    vlm_models = None if args.no_vlm else args.vlm_models
    if vlm_models:
        print(f"VLM Processing: Enabled ({', '.join(vlm_models)})")
    else:
        print(f"VLM Processing: Disabled")
    print()
    
    # Initialize processor
    processor = EnhancedDebugDocumentProcessor(debug_config, vlm_models)
    
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
        results = await processor.process_batch_with_enhanced_debug(files_to_process)
        
        # Print summary
        successful = sum(1 for r in results if "error" not in r)
        print(f"\n📊 Processing Summary:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
        # Show enhanced reports
        print(f"\n📂 Enhanced reports saved to: {debug_config.output_dir}")
        for result in results:
            if result.get("enhanced_report"):
                print(f"   - {Path(result['enhanced_report']).name}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())