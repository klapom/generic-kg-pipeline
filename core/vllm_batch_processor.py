"""
Enhanced Batch Processor with vLLM Model Lifecycle Management

Provides complete batch processing workflow:
Loading → Warmup → Processing → Cleanup

Integrates with VLLMModelManager for optimal performance and resource management.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.vllm.model_manager import VLLMModelManager, model_manager
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.clients.vllm_qwen25_vl_local import VLLMQwen25VLClient
from core.content_chunker import ContentChunker
from plugins.parsers.parser_factory import ParserFactory
from plugins.parsers.base_parser import Document, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    use_vllm: bool = True
    max_concurrent: int = 3
    timeout_seconds: int = 600
    enable_chunking: bool = True
    enable_context_inheritance: bool = True
    gpu_memory_utilization: float = 0.8
    
    # Model-specific settings
    smoldocling_settings: Dict[str, Any] = None
    qwen25_vl_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.smoldocling_settings is None:
            self.smoldocling_settings = {
                "max_pages": 100,
                "extract_tables": True,
                "extract_images": True,
                "extract_formulas": True,
                "preserve_layout": True
            }
        
        if self.qwen25_vl_settings is None:
            self.qwen25_vl_settings = {
                "max_image_size": 1024,
                "image_quality": 85,
                "batch_size": 3
            }


@dataclass
class BatchProcessingResult:
    """Result from batch processing"""
    file_path: Path
    success: bool
    document: Optional[Document] = None
    chunks: Optional[List[Any]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchStats:
    """Statistics for batch processing"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_processing_time: float = 0.0
    model_load_time: float = 0.0
    warmup_time: float = 0.0
    cleanup_time: float = 0.0
    documents_processed: int = 0
    chunks_generated: int = 0
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100
    
    @property
    def average_processing_time(self) -> float:
        """Average processing time per file"""
        if self.successful_files == 0:
            return 0.0
        return self.total_processing_time / self.successful_files


class VLLMBatchProcessor:
    """
    Enhanced Batch Processor with complete vLLM Model Lifecycle
    
    Provides comprehensive document processing with:
    - Model loading and warmup
    - Batch document processing
    - Automatic cleanup and resource management
    - Performance monitoring and statistics
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.model_manager = model_manager
        self.parser_factory = ParserFactory()
        self.stats = BatchStats()
        
        # Initialize clients
        self.smoldocling_client: Optional[VLLMSmolDoclingClient] = None
        self.qwen25_vl_client: Optional[VLLMQwen25VLClient] = None
        self.content_chunker: Optional[ContentChunker] = None
        
        self._initialized = False
        
        logger.info(f"Initialized VLLMBatchProcessor (vLLM: {config.use_vllm})")
    
    def _create_clients(self):
        """Create and configure vLLM clients"""
        try:
            if not self.config.use_vllm:
                raise ValueError("vLLM must be enabled. Mock mode is no longer supported.")
                
            logger.info("Creating vLLM clients")
            
            # Create vLLM clients
            self.smoldocling_client = VLLMSmolDoclingClient(
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                **self.config.smoldocling_settings
            )
            
            self.qwen25_vl_client = VLLMQwen25VLClient(
                gpu_memory_utilization=self.config.gpu_memory_utilization * 0.8,  # Share GPU
                **self.config.qwen25_vl_settings
            )
            
            # Create content chunker if enabled
            if self.config.enable_chunking:
                chunking_config = {
                    "chunking": {
                        "strategies": {
                            "pdf": {
                                "max_tokens": 500,
                                "min_tokens": 100,
                                "overlap_tokens": 50,
                                "respect_boundaries": True
                            },
                            "txt": {
                                "max_tokens": 500,
                                "min_tokens": 100,
                                "overlap_tokens": 50,
                                "respect_boundaries": True
                            }
                        },
                        "enable_context_inheritance": self.config.enable_context_inheritance,
                        "context_inheritance": {
                            "enabled": self.config.enable_context_inheritance,
                            "max_context_tokens": 300,
                            "llm": {
                                "model": "hochschul-llm",
                                "temperature": 0.1
                            }
                        },
                        "performance": {
                            "enable_async_processing": True,
                            "max_concurrent_groups": self.config.max_concurrent
                        }
                    }
                }
                
                self.content_chunker = ContentChunker(chunking_config)
            
            self._initialized = True
            logger.info("✅ Clients created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create clients: {e}")
            raise
    
    def load_models(self) -> bool:
        """
        Phase 1: Load all required models
        
        Returns:
            True if all models loaded successfully
        """
        logger.info("🔧 Phase 1: Loading vLLM models...")
        start_time = time.time()
        
        try:
            if not self.config.use_vllm:
                raise RuntimeError("vLLM must be enabled. Mock mode is no longer supported.")
            
            # Check vLLM availability
            if not self.model_manager.check_vllm_availability():
                raise RuntimeError("vLLM not available. Please ensure vLLM is installed and GPU is accessible.")
            
            loaded_models = []
            
            # Load SmolDocling model
            if self.smoldocling_client:
                logger.info("Loading SmolDocling model...")
                if self.smoldocling_client.ensure_model_loaded():
                    loaded_models.append("smoldocling")
                    logger.info("✅ SmolDocling model loaded")
                else:
                    logger.error("❌ Failed to load SmolDocling model")
            
            # Load Qwen2.5-VL model
            if self.qwen25_vl_client:
                logger.info("Loading Qwen2.5-VL model...")
                if self.qwen25_vl_client.ensure_model_loaded():
                    loaded_models.append("qwen25_vl")
                    logger.info("✅ Qwen2.5-VL model loaded")
                else:
                    logger.error("❌ Failed to load Qwen2.5-VL model")
            
            load_time = time.time() - start_time
            self.stats.model_load_time = load_time
            
            logger.info(f"✅ Phase 1 completed: {len(loaded_models)} models loaded in {load_time:.1f}s")
            
            # Print GPU memory info
            gpu_info = self.model_manager.get_gpu_memory_info()
            if gpu_info["available"]:
                for device in gpu_info["devices"]:
                    util = device["utilization"]
                    logger.info(f"   GPU {device['device_id']}: {util:.1%} utilized")
            
            return len(loaded_models) > 0
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.stats.model_load_time = time.time() - start_time
            return False
    
    def warmup_models(self) -> bool:
        """
        Phase 2: Warm up all loaded models
        
        Returns:
            True if warmup successful
        """
        logger.info("🔥 Phase 2: Warming up models...")
        start_time = time.time()
        
        try:
            if not self.config.use_vllm:
                raise RuntimeError("vLLM must be enabled. Mock mode is no longer supported.")
            
            warmup_success = []
            
            # Warmup SmolDocling
            if self.smoldocling_client and hasattr(self.smoldocling_client, 'model_manager'):
                logger.info("Warming up SmolDocling...")
                if self.model_manager.warmup_model("smoldocling"):
                    warmup_success.append("smoldocling")
                    logger.info("✅ SmolDocling warmed up")
            
            # Warmup Qwen2.5-VL
            if self.qwen25_vl_client and hasattr(self.qwen25_vl_client, 'model_manager'):
                logger.info("Warming up Qwen2.5-VL...")
                if self.model_manager.warmup_model("qwen25_vl"):
                    warmup_success.append("qwen25_vl")
                    logger.info("✅ Qwen2.5-VL warmed up")
            
            warmup_time = time.time() - start_time
            self.stats.warmup_time = warmup_time
            
            logger.info(f"✅ Phase 2 completed: {len(warmup_success)} models warmed up in {warmup_time:.1f}s")
            
            return len(warmup_success) > 0
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            self.stats.warmup_time = time.time() - start_time
            return False
    
    async def process_single_file(self, file_path: Path) -> BatchProcessingResult:
        """
        Process a single document file
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            Processing result with document and chunks
        """
        start_time = time.time()
        
        logger.info(f"📄 Processing: {file_path.name}")
        
        try:
            # Get appropriate parser
            parser = self.parser_factory.get_parser_for_file(file_path)
            
            # Parse document
            if file_path.suffix.lower() == '.pdf' and self.smoldocling_client:
                # Use vLLM SmolDocling for PDF
                if hasattr(self.smoldocling_client, 'parse_pdf'):
                    result = self.smoldocling_client.parse_pdf(file_path)
                    document = self.smoldocling_client.convert_to_document(result, file_path)
                else:
                    # Fallback for mock client
                    document = await parser.parse_document(file_path)
            else:
                # Use standard parser for other formats
                document = await parser.parse_document(file_path)
            
            # Process visual elements with Qwen2.5-VL if available
            if document.visual_elements and self.qwen25_vl_client:
                logger.debug(f"Analyzing {len(document.visual_elements)} visual elements")
                # Here you could add visual element processing
                # For now, we'll skip this to keep the implementation focused
            
            chunks = None
            if self.config.enable_chunking and self.content_chunker:
                logger.debug("Chunking document...")
                chunking_result = await self.content_chunker.chunk_document(document)
                chunks = chunking_result.contextual_chunks
                self.stats.chunks_generated += len(chunks)
            
            processing_time = time.time() - start_time
            
            result = BatchProcessingResult(
                file_path=file_path,
                success=True,
                document=document,
                chunks=chunks,
                processing_time=processing_time,
                metadata={
                    "document_type": document.metadata.document_type.value,
                    "page_count": document.metadata.page_count,
                    "segments_count": len(document.segments),
                    "visual_elements_count": len(document.visual_elements) if document.visual_elements else 0,
                    "chunks_count": len(chunks) if chunks else 0
                }
            )
            
            self.stats.successful_files += 1
            self.stats.documents_processed += 1
            self.stats.total_processing_time += processing_time
            
            logger.info(f"✅ {file_path.name}: {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing failed: {str(e)}"
            
            logger.error(f"❌ {file_path.name}: {error_msg}")
            
            self.stats.failed_files += 1
            
            return BatchProcessingResult(
                file_path=file_path,
                success=False,
                processing_time=processing_time,
                error_message=error_msg,
                metadata={"error": str(e)}
            )
    
    async def process_batch(self, file_paths: List[Path]) -> List[BatchProcessingResult]:
        """
        Phase 3: Process a batch of documents
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            List of processing results
        """
        if not file_paths:
            return []
        
        logger.info(f"📊 Phase 3: Processing batch of {len(file_paths)} files...")
        
        self.stats.total_files = len(file_paths)
        batch_start = time.time()
        
        # Process with controlled concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(file_path: Path) -> BatchProcessingResult:
            async with semaphore:
                return await self.process_single_file(file_path)
        
        # Create tasks for all files
        tasks = [process_with_semaphore(file_path) for file_path in file_paths]
        
        # Process with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            # Log progress
            progress = ((i + 1) / len(file_paths)) * 100
            logger.info(f"   Progress: {i + 1}/{len(file_paths)} ({progress:.1f}%)")
        
        batch_time = time.time() - batch_start
        
        logger.info(f"✅ Phase 3 completed in {batch_time:.1f}s")
        logger.info(f"   Success rate: {self.stats.success_rate:.1f}%")
        logger.info(f"   Average per file: {batch_time/len(file_paths):.1f}s")
        
        return results
    
    def cleanup_models(self) -> bool:
        """
        Phase 4: Cleanup and unload models
        
        Returns:
            True if cleanup successful
        """
        logger.info("🧹 Phase 4: Cleaning up models...")
        start_time = time.time()
        
        try:
            if not self.config.use_vllm:
                raise RuntimeError("vLLM must be enabled. Mock mode is no longer supported.")
            
            # Cleanup clients
            if self.smoldocling_client and hasattr(self.smoldocling_client, 'cleanup'):
                self.smoldocling_client.cleanup()
            
            if self.qwen25_vl_client and hasattr(self.qwen25_vl_client, 'cleanup'):
                self.qwen25_vl_client.cleanup()
            
            # Unload all models from manager
            self.model_manager.unload_all_models()
            
            cleanup_time = time.time() - start_time
            self.stats.cleanup_time = cleanup_time
            
            logger.info(f"✅ Phase 4 completed: Cleanup done in {cleanup_time:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.stats.cleanup_time = time.time() - start_time
            return False
    
    def print_statistics(self):
        """Print comprehensive batch processing statistics"""
        logger.info("📊 Batch Processing Statistics:")
        logger.info(f"   Files processed: {self.stats.total_files}")
        logger.info(f"   Successful: {self.stats.successful_files}")
        logger.info(f"   Failed: {self.stats.failed_files}")
        logger.info(f"   Success rate: {self.stats.success_rate:.1f}%")
        logger.info(f"   Documents processed: {self.stats.documents_processed}")
        logger.info(f"   Chunks generated: {self.stats.chunks_generated}")
        logger.info("")
        logger.info("📈 Performance Metrics:")
        logger.info(f"   Model load time: {self.stats.model_load_time:.1f}s")
        logger.info(f"   Warmup time: {self.stats.warmup_time:.1f}s")
        logger.info(f"   Total processing time: {self.stats.total_processing_time:.1f}s")
        logger.info(f"   Cleanup time: {self.stats.cleanup_time:.1f}s")
        logger.info(f"   Average per file: {self.stats.average_processing_time:.1f}s")
        
        if self.stats.model_load_time > 0 and self.stats.successful_files > 1:
            time_saved = self.stats.model_load_time * (self.stats.successful_files - 1)
            logger.info(f"   Time saved by caching: {time_saved:.1f}s")
        
        # Print model manager statistics
        self.model_manager.print_statistics()
    
    async def run_complete_batch(self, file_paths: List[Path]) -> List[BatchProcessingResult]:
        """
        Run complete batch processing workflow
        
        Args:
            file_paths: List of document file paths to process
            
        Returns:
            List of processing results
        """
        if not file_paths:
            logger.warning("No files provided for batch processing")
            return []
        
        logger.info(f"🚀 Starting complete batch processing workflow: {len(file_paths)} files")
        
        try:
            # Initialize clients
            if not self._initialized:
                self._create_clients()
            
            # Phase 1: Load models
            if not self.load_models():
                logger.error("Failed to load models - aborting batch processing")
                return []
            
            # Phase 2: Warmup
            if not self.warmup_models():
                logger.warning("Model warmup failed - continuing anyway")
            
            # Phase 3: Process documents
            results = await self.process_batch(file_paths)
            
            return results
            
        finally:
            # Phase 4: Cleanup (always executed)
            self.cleanup_models()
            
            # Print final statistics
            self.print_statistics()
    
    @asynccontextmanager
    async def batch_context(self, file_paths: List[Path]):
        """
        Async context manager for batch processing
        
        Args:
            file_paths: Files to process in this batch context
        """
        try:
            # Setup
            if not self._initialized:
                self._create_clients()
            
            self.load_models()
            self.warmup_models()
            
            logger.info(f"🚀 Batch context ready for {len(file_paths)} files")
            yield self
            
        finally:
            # Cleanup
            self.cleanup_models()
            logger.info("🏁 Batch context cleanup completed")


async def run_vllm_batch_processing(
    input_directory: Path,
    output_directory: Path,
    config: Optional[BatchProcessingConfig] = None
) -> List[BatchProcessingResult]:
    """
    Convenience function for complete batch processing
    
    Args:
        input_directory: Directory containing documents to process
        output_directory: Directory for output files
        config: Optional batch processing configuration
        
    Returns:
        List of processing results
    """
    if config is None:
        config = BatchProcessingConfig()
    
    # Find supported files
    supported_extensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt']
    file_paths = []
    
    for ext in supported_extensions:
        file_paths.extend(input_directory.glob(f"*{ext}"))
    
    if not file_paths:
        logger.warning(f"No supported files found in {input_directory}")
        return []
    
    file_paths = sorted(file_paths)
    
    # Create processor and run
    processor = VLLMBatchProcessor(config)
    results = await processor.run_complete_batch(file_paths)
    
    # Save results to output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    import json
    for result in results:
        if result.success:
            output_file = output_directory / f"{result.file_path.stem}_processed.json"
            
            output_data = {
                "file_info": {
                    "filename": result.file_path.name,
                    "file_path": str(result.file_path),
                    "processed_at": datetime.now().isoformat()
                },
                "processing": {
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata
                },
                "document": {
                    "title": result.document.metadata.title if result.document else None,
                    "content_length": len(result.document.content) if result.document else 0,
                    "segments_count": len(result.document.segments) if result.document else 0
                },
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "token_count": chunk.token_count
                    }
                    for chunk in (result.chunks or [])
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return results