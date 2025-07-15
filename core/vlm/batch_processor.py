#!/usr/bin/env python3
"""
Batch Document Processor for efficient multi-document VLM analysis.
Optimizes processing of multiple PDFs with intelligent batching.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, field

from core.parsers.hybrid_pdf_parser import HybridPDFParser
from core.vlm.two_stage_processor import TwoStageVLMProcessor
from core.parsers.interfaces.data_models import VisualElement

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 8
    max_memory_gb: float = 14.0  # Leave headroom
    enable_parallel_extraction: bool = True
    max_workers: int = field(default_factory=lambda: min(4, mp.cpu_count()))
    confidence_threshold: float = 0.85


@dataclass
class BatchResult:
    """Result of batch processing."""
    document_path: Path
    success: bool
    visual_count: int
    processing_time: float
    results: List[Any] = field(default_factory=list)
    error: Optional[str] = None


class BatchDocumentProcessor:
    """
    Handles batch processing of multiple documents with optimal resource usage.
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchProcessingConfig()
        self.pdf_parser = HybridPDFParser()
        self.vlm_processor = TwoStageVLMProcessor(
            confidence_threshold=self.config.confidence_threshold
        )
        self.processing_stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "total_visuals": 0,
            "total_time": 0,
            "batch_times": []
        }
    
    async def process_documents(self, pdf_paths: List[Union[str, Path]]) -> List[BatchResult]:
        """
        Process multiple PDF documents in batches.
        
        Args:
            pdf_paths: List of paths to PDF documents
            
        Returns:
            List of batch results
        """
        start_time = time.time()
        pdf_paths = [Path(p) for p in pdf_paths]
        self.processing_stats["total_documents"] = len(pdf_paths)
        
        logger.info(f"Starting batch processing of {len(pdf_paths)} documents")
        
        # Extract visuals from all documents
        all_visuals = await self._extract_all_visuals(pdf_paths)
        
        # Process visuals in optimized batches
        results = await self._process_visual_batches(all_visuals)
        
        # Update stats
        self.processing_stats["total_time"] = time.time() - start_time
        self._log_stats()
        
        return results
    
    async def _extract_all_visuals(self, pdf_paths: List[Path]) -> List[Tuple[Path, List[VisualElement]]]:
        """
        Extract visuals from all PDFs, optionally in parallel.
        
        Args:
            pdf_paths: List of PDF paths
            
        Returns:
            List of (path, visuals) tuples
        """
        logger.info("Extracting visuals from all documents...")
        all_visuals = []
        
        if self.config.enable_parallel_extraction:
            # Parallel extraction
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all extraction tasks
                future_to_path = {
                    executor.submit(self._extract_visuals_sync, path): path 
                    for path in pdf_paths
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        visuals = future.result()
                        all_visuals.append((path, visuals))
                        logger.info(f"Extracted {len(visuals)} visuals from {path.name}")
                    except Exception as e:
                        logger.error(f"Failed to extract from {path.name}: {e}")
                        all_visuals.append((path, []))
        else:
            # Sequential extraction
            for path in pdf_paths:
                try:
                    visuals = await self._extract_visuals_async(path)
                    all_visuals.append((path, visuals))
                    logger.info(f"Extracted {len(visuals)} visuals from {path.name}")
                except Exception as e:
                    logger.error(f"Failed to extract from {path.name}: {e}")
                    all_visuals.append((path, []))
        
        # Count total visuals
        total_visuals = sum(len(visuals) for _, visuals in all_visuals)
        self.processing_stats["total_visuals"] = total_visuals
        logger.info(f"Total visuals extracted: {total_visuals}")
        
        return all_visuals
    
    def _extract_visuals_sync(self, pdf_path: Path) -> List[VisualElement]:
        """
        Synchronous visual extraction for thread pool.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            List of visual elements
        """
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._extract_visuals_async(pdf_path))
        finally:
            loop.close()
    
    async def _extract_visuals_async(self, pdf_path: Path) -> List[VisualElement]:
        """
        Extract visuals from a single PDF.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            List of visual elements
        """
        parsed_doc = await self.pdf_parser.parse(pdf_path)
        # Return visual_elements directly from the document
        return parsed_doc.visual_elements
    
    async def _process_visual_batches(self, 
                                    all_visuals: List[Tuple[Path, List[VisualElement]]]) -> List[BatchResult]:
        """
        Process visuals in optimized batches.
        
        Args:
            all_visuals: List of (path, visuals) tuples
            
        Returns:
            List of batch results
        """
        results = []
        
        # Create optimal batches
        batches = self._create_optimal_batches(all_visuals)
        
        logger.info(f"Processing {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            
            # Flatten visuals for batch processing
            batch_visuals = []
            visual_to_doc = {}  # Map visual to document
            
            for doc_path, visuals in batch:
                for visual in visuals:
                    batch_visuals.append(visual)
                    visual_to_doc[visual.content_hash] = doc_path
            
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} "
                       f"with {len(batch_visuals)} visuals")
            
            # Process batch with VLM
            try:
                vlm_results = self.vlm_processor.process_batch(batch_visuals)
                
                # Group results by document
                doc_results = {}
                for visual, result in zip(batch_visuals, vlm_results):
                    doc_path = visual_to_doc[visual.content_hash]
                    if doc_path not in doc_results:
                        doc_results[doc_path] = []
                    doc_results[doc_path].append(result)
                
                # Create batch results
                for doc_path, doc_visuals in batch:
                    if doc_path in doc_results:
                        results.append(BatchResult(
                            document_path=doc_path,
                            success=True,
                            visual_count=len(doc_visuals),
                            processing_time=time.time() - batch_start,
                            results=doc_results[doc_path]
                        ))
                        self.processing_stats["successful_documents"] += 1
                    else:
                        results.append(BatchResult(
                            document_path=doc_path,
                            success=False,
                            visual_count=len(doc_visuals),
                            processing_time=time.time() - batch_start,
                            error="No results returned"
                        ))
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                # Mark all documents in batch as failed
                for doc_path, doc_visuals in batch:
                    results.append(BatchResult(
                        document_path=doc_path,
                        success=False,
                        visual_count=len(doc_visuals),
                        processing_time=time.time() - batch_start,
                        error=str(e)
                    ))
            
            batch_time = time.time() - batch_start
            self.processing_stats["batch_times"].append(batch_time)
            logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.1f}s")
        
        return results
    
    def _create_optimal_batches(self, 
                               all_visuals: List[Tuple[Path, List[VisualElement]]]) -> List[List[Tuple[Path, List[VisualElement]]]]:
        """
        Create optimal batches based on memory and processing constraints.
        
        Args:
            all_visuals: List of (path, visuals) tuples
            
        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        current_size = 0
        
        # Estimate memory per visual (rough estimate: 50MB per image)
        memory_per_visual = 0.05  # GB
        max_visuals_per_batch = int(self.config.max_memory_gb / memory_per_visual)
        max_visuals_per_batch = min(max_visuals_per_batch, self.config.max_batch_size)
        
        for doc_path, visuals in all_visuals:
            doc_size = len(visuals)
            
            # Check if adding this document exceeds batch limits
            if current_size + doc_size > max_visuals_per_batch and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            # Add document to current batch
            current_batch.append((doc_path, visuals))
            current_size += doc_size
            
            # Check if batch is full
            if current_size >= max_visuals_per_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
        
        # Add remaining documents
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches with max {max_visuals_per_batch} visuals per batch")
        
        return batches
    
    def _log_stats(self):
        """Log processing statistics."""
        stats = self.processing_stats
        avg_batch_time = sum(stats["batch_times"]) / len(stats["batch_times"]) if stats["batch_times"] else 0
        
        logger.info(f"""
Batch Processing Complete:
- Total documents: {stats['total_documents']}
- Successful documents: {stats['successful_documents']}
- Total visuals processed: {stats['total_visuals']}
- Total time: {stats['total_time']:.1f}s
- Average time per document: {stats['total_time'] / max(stats['total_documents'], 1):.1f}s
- Average time per visual: {stats['total_time'] / max(stats['total_visuals'], 1):.1f}s
- Number of batches: {len(stats['batch_times'])}
- Average batch time: {avg_batch_time:.1f}s
""")
    
    def cleanup(self):
        """Clean up resources."""
        self.vlm_processor.cleanup()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "max_batch_size": self.config.max_batch_size,
            "max_memory_gb": self.config.max_memory_gb,
            "parallel_extraction": self.config.enable_parallel_extraction,
            "max_workers": self.config.max_workers,
            "confidence_threshold": self.config.confidence_threshold
        }