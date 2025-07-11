"""Main content chunker with context inheritance support"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from plugins.parsers.base_parser import Document, DocumentType
from core.config import load_chunking_config
from core.chunking import (
    ContextualChunk,
    ContextGroup,
    ChunkingResult,
    ChunkingStats,
    ChunkingStrategy,
    BaseChunker,
    ContextGrouper,
    ContextSummarizer
)

logger = logging.getLogger(__name__)


class ContentChunker:
    """
    Main content chunker with context inheritance
    
    Orchestrates the entire chunking process:
    1. Structure-aware chunking
    2. Context group formation
    3. Context inheritance processing
    4. Quality validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize content chunker"""
        
        # Load configuration
        if config is None:
            config = load_chunking_config()
        
        self.config = config
        self.chunking_config = config.get("chunking", {})
        
        # Initialize components
        self.base_chunker = StructureAwareChunker(config)
        self.context_grouper = ContextGrouper(config)
        self.context_summarizer = ContextSummarizer(config)
        
        # Settings
        self.enable_context_inheritance = self.chunking_config.get("enable_context_inheritance", True)
        self.enable_async_processing = self.chunking_config.get("performance", {}).get("enable_async_processing", True)
        self.max_concurrent_groups = self.chunking_config.get("performance", {}).get("max_concurrent_groups", 5)
        
        logger.info(f"Initialized ContentChunker with context inheritance: {self.enable_context_inheritance}")
    
    async def chunk_document(
        self,
        document: Document,
        task_template: Optional[str] = None
    ) -> ChunkingResult:
        """
        Chunk document with context inheritance
        
        Args:
            document: Document to chunk
            task_template: Template for the main task (for context generation)
            
        Returns:
            ChunkingResult with contextual chunks and groups
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting chunking for document: {document.metadata.title}")
            
            # Get document-specific configuration
            doc_type = document.metadata.document_type.value
            strategy_config = self.chunking_config.get("strategies", {}).get(doc_type, {})
            
            # Step 1: Create base chunks using structure-aware strategy
            logger.info("🔧 STEP 1: Creating base chunks with structure-aware strategy...")
            base_chunks = self.base_chunker.chunk_document(document, strategy_config)
            
            if not base_chunks:
                logger.warning(f"❌ No chunks created for document {document.metadata.title}")
                return self._create_empty_result(document)
            
            logger.info(f"✅ Created {len(base_chunks)} base chunks")
            
            # Log details of each base chunk
            for i, chunk in enumerate(base_chunks, 1):
                logger.info(f"📄 Base Chunk {i}: {chunk.chunk_type}, {chunk.token_count} tokens, {len(chunk.content)} chars")
            
            # Step 2: Group chunks into context units
            logger.info("🔧 STEP 2: Grouping chunks into context units...")
            context_groups = self.context_grouper.group_chunks(base_chunks, document)
            
            if not context_groups:
                logger.warning(f"❌ No context groups created for document {document.metadata.title}")
                context_groups = [self._create_fallback_group(base_chunks, document)]
            
            logger.info(f"✅ Created {len(context_groups)} context groups")
            
            # Log details of each context group
            for i, group in enumerate(context_groups, 1):
                logger.info(f"📦 Context Group {i}: {len(group.chunks)} chunks, document: {group.document_id}")
            
            # Step 3: Process context inheritance
            if self.enable_context_inheritance and task_template:
                logger.info("🔧 STEP 3: Processing context inheritance (LOCAL Hochschul-LLM)...")
                context_groups = await self._process_context_inheritance(
                    context_groups, 
                    task_template
                )
                logger.info("✅ Context inheritance processing completed")
            else:
                logger.info("⏭️ STEP 3: Context inheritance disabled - skipping")
            
            # Step 4: Collect all processed chunks
            logger.info("🔧 STEP 4: Collecting all processed chunks...")
            all_chunks = []
            for group in context_groups:
                all_chunks.extend(group.chunks)
            
            logger.info(f"✅ Collected {len(all_chunks)} final chunks")
            
            # Log final chunk details
            for i, chunk in enumerate(all_chunks, 1):
                has_context = "✅" if chunk.inherited_context else "❌"
                generates_context = "✅" if chunk.generates_context else "❌"
                logger.info(f"📄 Final Chunk {i}: {chunk.chunk_type}, {chunk.token_count} tokens, has_context: {has_context}, generates_context: {generates_context}")
            
            # Step 5: Validate chunks
            logger.info("🔧 STEP 5: Validating chunks...")
            all_chunks = self.base_chunker.validate_chunks(all_chunks, strategy_config)
            logger.info(f"✅ Validation completed - {len(all_chunks)} chunks validated")
            
            # Step 6: Create processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            stats = self.base_chunker.create_chunking_stats(
                chunks=all_chunks,
                processing_time=processing_time,
                context_generation_time=0.0  # Would be calculated from context processing
            )
            
            # Step 7: Create final result
            result = ChunkingResult(
                document_id=document.metadata.title or "unknown",
                source_document=document,
                contextual_chunks=all_chunks,
                context_groups=context_groups,
                chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
                processing_stats=stats,
                processing_config=strategy_config
            )
            
            logger.info(f"Chunking completed: {result.total_chunks} chunks in {result.total_groups} groups")
            
            return result
            
        except Exception as e:
            logger.error(f"Chunking failed for document {document.metadata.title}: {e}")
            raise
    
    async def chunk_document_with_context_inheritance(
        self,
        document: Document,
        task_template: str
    ) -> ChunkingResult:
        """
        Chunk document with explicit context inheritance
        
        Args:
            document: Document to chunk
            task_template: Template for the main task (required for context generation)
            
        Returns:
            ChunkingResult with context inheritance applied
        """
        return await self.chunk_document(document, task_template)
    
    async def _process_context_inheritance(
        self,
        context_groups: List[ContextGroup],
        task_template: str
    ) -> List[ContextGroup]:
        """Process context inheritance for all groups"""
        
        if not context_groups:
            return context_groups
        
        try:
            if self.enable_async_processing:
                # Process groups concurrently
                processed_groups = await self._process_groups_async(context_groups, task_template)
            else:
                # Process groups sequentially
                processed_groups = await self._process_groups_sequential(context_groups, task_template)
            
            return processed_groups
            
        except Exception as e:
            logger.error(f"Context inheritance processing failed: {e}")
            return context_groups
    
    async def _process_groups_async(
        self,
        context_groups: List[ContextGroup],
        task_template: str
    ) -> List[ContextGroup]:
        """Process context groups asynchronously"""
        
        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.max_concurrent_groups)
        
        async def process_single_group(group: ContextGroup) -> ContextGroup:
            async with semaphore:
                return await self._process_single_context_group(group, task_template)
        
        # Process all groups concurrently
        tasks = [process_single_group(group) for group in context_groups]
        processed_groups = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        result_groups = []
        for i, result in enumerate(processed_groups):
            if isinstance(result, Exception):
                logger.error(f"Group processing failed for group {context_groups[i].group_id}: {result}")
                result_groups.append(context_groups[i])  # Use original group
            else:
                result_groups.append(result)
        
        return result_groups
    
    async def _process_groups_sequential(
        self,
        context_groups: List[ContextGroup],
        task_template: str
    ) -> List[ContextGroup]:
        """Process context groups sequentially"""
        
        processed_groups = []
        
        for group in context_groups:
            try:
                processed_group = await self._process_single_context_group(group, task_template)
                processed_groups.append(processed_group)
            except Exception as e:
                logger.error(f"Group processing failed for group {group.group_id}: {e}")
                processed_groups.append(group)  # Use original group
        
        return processed_groups
    
    async def _process_single_context_group(
        self,
        group: ContextGroup,
        task_template: str
    ) -> ContextGroup:
        """Process context inheritance for a single group"""
        
        try:
            # Skip groups with only one chunk
            if len(group.chunks) <= 1:
                return group
            
            logger.debug(f"Processing context inheritance for group {group.group_id} with {len(group.chunks)} chunks")
            
            # Process context inheritance
            processed_chunks = await self.context_summarizer.process_context_inheritance(
                chunks=group.chunks,
                task_template=task_template,
                group_id=group.group_id
            )
            
            # Update group with processed chunks
            group.chunks = processed_chunks
            
            # Update group context summary from first chunk
            if processed_chunks and processed_chunks[0].inherited_context:
                group.context_summary = processed_chunks[0].inherited_context
                group.context_generated_at = datetime.now()
            
            return group
            
        except Exception as e:
            logger.error(f"Context inheritance processing failed for group {group.group_id}: {e}")
            return group
    
    def _create_empty_result(self, document: Document) -> ChunkingResult:
        """Create empty result when no chunks are created"""
        return ChunkingResult(
            document_id=document.metadata.title or "unknown",
            source_document=document,
            contextual_chunks=[],
            context_groups=[],
            chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
            processing_stats=ChunkingStats(),
            processing_config={}
        )
    
    def _create_fallback_group(self, chunks: List[ContextualChunk], document: Document) -> ContextGroup:
        """Create fallback group when grouping fails"""
        from core.chunking.chunk_models import ContextGroupType
        
        group = ContextGroup(
            group_id=f"fallback_{document.metadata.title}",
            document_id=document.metadata.title or "unknown",
            group_type=ContextGroupType.PAGE_RANGE,
            chunks=chunks,
            group_metadata={"fallback": True}
        )
        
        return group
    
    async def chunk_multiple_documents(
        self,
        documents: List[Document],
        task_template: Optional[str] = None
    ) -> List[ChunkingResult]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of documents to chunk
            task_template: Template for the main task
            
        Returns:
            List of chunking results
        """
        results = []
        
        for document in documents:
            try:
                result = await self.chunk_document(document, task_template)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to chunk document {document.metadata.title}: {e}")
                # Create empty result for failed document
                empty_result = self._create_empty_result(document)
                results.append(empty_result)
        
        return results
    
    def get_chunking_stats(self, results: List[ChunkingResult]) -> Dict[str, Any]:
        """Get aggregated statistics from multiple chunking results"""
        
        total_chunks = sum(result.total_chunks for result in results)
        total_groups = sum(result.total_groups for result in results)
        chunks_with_context = sum(result.chunks_with_context for result in results)
        
        # Calculate averages
        avg_chunks_per_doc = total_chunks / len(results) if results else 0
        avg_groups_per_doc = total_groups / len(results) if results else 0
        context_coverage = chunks_with_context / total_chunks if total_chunks > 0 else 0
        
        # Processing times
        total_processing_time = sum(
            result.processing_stats.processing_time_seconds 
            for result in results 
            if result.processing_stats
        )
        
        return {
            "total_documents": len(results),
            "total_chunks": total_chunks,
            "total_groups": total_groups,
            "chunks_with_context": chunks_with_context,
            "avg_chunks_per_doc": avg_chunks_per_doc,
            "avg_groups_per_doc": avg_groups_per_doc,
            "context_coverage_ratio": context_coverage,
            "total_processing_time": total_processing_time,
            "avg_processing_time_per_doc": total_processing_time / len(results) if results else 0
        }


class StructureAwareChunker(BaseChunker):
    """Concrete implementation of structure-aware chunking"""
    
    def chunk_document(self, document: Document, strategy_config: Dict[str, Any]) -> List[ContextualChunk]:
        """
        Chunk document using structure-aware strategy
        
        Args:
            document: Document to chunk
            strategy_config: Configuration for chunking strategy
            
        Returns:
            List of contextual chunks
        """
        try:
            # Extract document structure information
            structure_info = self.extract_structure_info(document)
            
            # Convert segments to chunks
            chunks = self.segments_to_chunks(
                segments=document.segments,
                document_id=document.metadata.title or "unknown",
                strategy_config=strategy_config
            )
            
            # Integrate visual elements
            chunks = self._integrate_visual_elements(chunks, document)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Structure-aware chunking failed: {e}")
            raise
    
    def _integrate_visual_elements(self, chunks: List[ContextualChunk], document: Document) -> List[ContextualChunk]:
        """Integrate visual elements into chunks"""
        
        # Create mapping from pages to visual elements
        page_to_visuals = {}
        for visual in document.visual_elements:
            page = visual.page_or_slide
            if page:
                if page not in page_to_visuals:
                    page_to_visuals[page] = []
                page_to_visuals[page].append(visual)
        
        # Assign visual elements to chunks based on page ranges
        for chunk in chunks:
            if chunk.page_range:
                start_page, end_page = chunk.page_range
                chunk_visuals = []
                
                for page in range(start_page, end_page + 1):
                    if page in page_to_visuals:
                        chunk_visuals.extend(page_to_visuals[page])
                
                chunk.visual_elements = chunk_visuals
        
        return chunks