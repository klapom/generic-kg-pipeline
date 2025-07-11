"""Batch document processing from filesystem"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from plugins.parsers import ParserFactory, Document, DocumentType
from core.config import get_config

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Result of processing a single document"""
    file_path: Path
    status: ProcessingStatus
    document: Optional[Document] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": str(self.file_path),
            "status": self.status.value,
            "has_document": self.document is not None,
            "error": self.error,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "segments": self.document.total_segments if self.document else 0,
            "visual_elements": self.document.total_visual_elements if self.document else 0
        }


@dataclass
class BatchProcessingReport:
    """Report of batch processing operation"""
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    total_time: float
    results: List[ProcessingResult]
    start_time: datetime
    end_time: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "results": [r.to_dict() for r in self.results]
        }
    
    def save_report(self, output_path: Path):
        """Save report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Batch processing report saved to {output_path}")


class BatchProcessor:
    """
    Batch document processor for filesystem operations
    
    Features:
    - Process directories recursively
    - Filter by file types
    - Concurrent processing
    - Progress tracking
    - Error handling
    - Result persistence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize batch processor"""
        self.config = config or {}
        self.system_config = get_config()
        
        # Configuration
        self.config.setdefault("max_concurrent", 5)
        self.config.setdefault("recursive", True)
        self.config.setdefault("skip_hidden", True)
        self.config.setdefault("save_results", True)
        self.config.setdefault("result_format", "json")
        self.config.setdefault("enable_vlm", True)
        self.config.setdefault("file_extensions", None)  # None = all supported
        
        # Initialize parser factory
        self.parser_factory = ParserFactory(
            config=self.config.get("parser_config", {}),
            enable_vlm=self.config.get("enable_vlm", True)
        )
        
        # Processing state
        self.processing_queue: asyncio.Queue = None
        self.results: List[ProcessingResult] = []
        self.is_processing = False
        
        logger.info(f"Initialized batch processor with max_concurrent={self.config['max_concurrent']}")
    
    async def process_directory(
        self, 
        directory: Path,
        output_dir: Optional[Path] = None,
        file_pattern: Optional[str] = None
    ) -> BatchProcessingReport:
        """
        Process all documents in a directory
        
        Args:
            directory: Directory to process
            output_dir: Directory to save results (optional)
            file_pattern: Glob pattern for file filtering (e.g., "*.pdf")
            
        Returns:
            BatchProcessingReport with results
        """
        try:
            start_time = datetime.now()
            
            # Validate directory
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory}")
            
            if not directory.is_dir():
                raise ValueError(f"Path is not a directory: {directory}")
            
            # Collect files
            files = self._collect_files(directory, file_pattern)
            logger.info(f"Found {len(files)} files to process in {directory}")
            
            if not files:
                return BatchProcessingReport(
                    total_files=0,
                    processed_files=0,
                    failed_files=0,
                    skipped_files=0,
                    total_time=0,
                    results=[],
                    start_time=start_time,
                    end_time=datetime.now()
                )
            
            # Process files
            results = await self._process_files_batch(files, output_dir)
            
            # Generate report
            end_time = datetime.now()
            report = self._generate_report(results, start_time, end_time)
            
            # Save report if configured
            if self.config.get("save_results", True) and output_dir:
                report_path = output_dir / f"batch_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
                report.save_report(report_path)
            
            return report
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    async def process_file_list(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None
    ) -> BatchProcessingReport:
        """
        Process a specific list of files
        
        Args:
            file_paths: List of file paths to process
            output_dir: Directory to save results (optional)
            
        Returns:
            BatchProcessingReport with results
        """
        try:
            start_time = datetime.now()
            
            # Validate files
            valid_files = []
            for file_path in file_paths:
                if file_path.exists() and file_path.is_file():
                    valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping invalid file: {file_path}")
            
            logger.info(f"Processing {len(valid_files)} valid files out of {len(file_paths)} provided")
            
            # Process files
            results = await self._process_files_batch(valid_files, output_dir)
            
            # Generate report
            end_time = datetime.now()
            report = self._generate_report(results, start_time, end_time)
            
            # Save report if configured
            if self.config.get("save_results", True) and output_dir:
                report_path = output_dir / f"batch_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
                report.save_report(report_path)
            
            return report
            
        except Exception as e:
            logger.error(f"File list processing failed: {e}")
            raise
    
    def _collect_files(self, directory: Path, pattern: Optional[str] = None) -> List[Path]:
        """Collect files from directory based on configuration"""
        files = []
        
        # Get supported extensions
        if self.config.get("file_extensions"):
            extensions = set(self.config["file_extensions"])
        else:
            extensions = set(self.parser_factory.get_supported_extensions())
        
        # Recursive or single-level
        if self.config.get("recursive", True):
            glob_pattern = "**/*" if not pattern else f"**/{pattern}"
            file_iterator = directory.glob(glob_pattern)
        else:
            glob_pattern = "*" if not pattern else pattern
            file_iterator = directory.glob(glob_pattern)
        
        # Filter files
        for file_path in file_iterator:
            # Skip directories
            if not file_path.is_file():
                continue
            
            # Skip hidden files if configured
            if self.config.get("skip_hidden", True) and file_path.name.startswith('.'):
                continue
            
            # Check extension
            if file_path.suffix.lower() in extensions:
                # Check if parser can handle it
                if self.parser_factory.can_parse(file_path):
                    files.append(file_path)
                else:
                    logger.debug(f"No parser available for: {file_path}")
        
        # Sort files for consistent processing
        files.sort()
        
        return files
    
    async def _process_files_batch(
        self, 
        files: List[Path], 
        output_dir: Optional[Path] = None
    ) -> List[ProcessingResult]:
        """Process files in batch with concurrency control"""
        # Initialize processing queue
        self.processing_queue = asyncio.Queue()
        self.results = []
        
        # Add all files to queue
        for file_path in files:
            await self.processing_queue.put(file_path)
        
        # Create worker tasks
        workers = []
        max_concurrent = min(self.config.get("max_concurrent", 5), len(files))
        
        for i in range(max_concurrent):
            worker = asyncio.create_task(self._process_worker(i, output_dir))
            workers.append(worker)
        
        # Wait for all files to be processed
        await self.processing_queue.join()
        
        # Cancel workers
        for worker in workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        return self.results
    
    async def _process_worker(self, worker_id: int, output_dir: Optional[Path] = None):
        """Worker task for processing files"""
        logger.info(f"Worker {worker_id} started")
        
        try:
            while True:
                try:
                    # Get file from queue
                    file_path = await asyncio.wait_for(
                        self.processing_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Process file
                    result = await self._process_single_file(file_path, output_dir)
                    self.results.append(result)
                    
                    # Mark task as done
                    self.processing_queue.task_done()
                    
                    # Log progress
                    logger.info(f"Worker {worker_id}: Processed {file_path.name} - {result.status.value}")
                    
                except asyncio.TimeoutError:
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} stopped")
    
    async def _process_single_file(
        self, 
        file_path: Path, 
        output_dir: Optional[Path] = None
    ) -> ProcessingResult:
        """Process a single file"""
        start_time = datetime.now()
        
        try:
            # Parse document
            document = await self.parser_factory.parse_document(file_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save parsed document if output directory provided
            if output_dir and self.config.get("save_parsed_documents", True):
                await self._save_parsed_document(document, file_path, output_dir)
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.COMPLETED,
                document=document,
                processing_time=processing_time,
                metadata={
                    "parser_used": document.metadata.document_type.value,
                    "segments": document.total_segments,
                    "visual_elements": document.total_visual_elements,
                    "has_vlm_descriptions": any(ve.vlm_description for ve in document.visual_elements)
                }
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                file_path=file_path,
                status=ProcessingStatus.FAILED,
                error=str(e),
                processing_time=processing_time,
                metadata={"error_type": type(e).__name__}
            )
    
    async def _save_parsed_document(self, document: Document, source_path: Path, output_dir: Path):
        """Save parsed document to output directory"""
        try:
            # Create output subdirectory
            doc_output_dir = output_dir / "parsed_documents"
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_name = f"{source_path.stem}_parsed.json"
            output_path = doc_output_dir / output_name
            
            # Prepare document data
            doc_data = {
                "source_file": str(source_path),
                "parsed_at": datetime.now().isoformat(),
                "metadata": {
                    "title": document.metadata.title,
                    "author": document.metadata.author,
                    "page_count": document.metadata.page_count,
                    "document_type": document.metadata.document_type.value,
                    "custom": document.metadata.custom_metadata
                },
                "content": document.content,
                "segments": [
                    {
                        "content": seg.content,
                        "page_number": seg.page_number,
                        "segment_type": seg.segment_type,
                        "visual_references": seg.visual_references,
                        "metadata": seg.metadata
                    }
                    for seg in document.segments
                ],
                "visual_elements": [
                    ve.to_dict() for ve in document.visual_elements
                ],
                "analysis_summary": document.get_analysis_summary()
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved parsed document to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save parsed document: {e}")
    
    def _generate_report(
        self, 
        results: List[ProcessingResult], 
        start_time: datetime, 
        end_time: datetime
    ) -> BatchProcessingReport:
        """Generate batch processing report"""
        # Count statuses
        processed = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)
        
        # Calculate total time
        total_time = (end_time - start_time).total_seconds()
        
        return BatchProcessingReport(
            total_files=len(results),
            processed_files=processed,
            failed_files=failed,
            skipped_files=skipped,
            total_time=total_time,
            results=results,
            start_time=start_time,
            end_time=end_time
        )
    
    async def process_with_callback(
        self,
        files: List[Path],
        callback: callable,
        output_dir: Optional[Path] = None
    ) -> BatchProcessingReport:
        """
        Process files with a callback function for each completed file
        
        Args:
            files: List of files to process
            callback: Async function called with (file_path, document) for each success
            output_dir: Directory to save results
            
        Returns:
            BatchProcessingReport
        """
        start_time = datetime.now()
        results = []
        
        for file_path in files:
            result = await self._process_single_file(file_path, output_dir)
            results.append(result)
            
            # Call callback for successful processing
            if result.status == ProcessingStatus.COMPLETED and result.document:
                try:
                    await callback(file_path, result.document)
                except Exception as e:
                    logger.error(f"Callback failed for {file_path}: {e}")
        
        end_time = datetime.now()
        return self._generate_report(results, start_time, end_time)


# Convenience functions
async def process_directory(
    directory: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> BatchProcessingReport:
    """Process all documents in a directory"""
    processor = BatchProcessor(config)
    return await processor.process_directory(
        Path(directory),
        Path(output_dir) if output_dir else None
    )


async def process_files(
    file_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None
) -> BatchProcessingReport:
    """Process a list of specific files"""
    processor = BatchProcessor(config)
    paths = [Path(p) for p in file_paths]
    return await processor.process_file_list(
        paths,
        Path(output_dir) if output_dir else None
    )