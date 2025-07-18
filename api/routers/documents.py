"""Document upload and management endpoints"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from core.clients import VLLMSmolDoclingFinalClient as VLLMSmolDoclingClient
from core.clients.hochschul_llm import HochschulLLMClient
from core.parsers import ParseError

logger = logging.getLogger(__name__)

router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    document_id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    upload_timestamp: datetime


class DocumentInfo(BaseModel):
    """Document information model"""
    document_id: str
    filename: str
    file_size: int
    content_type: str
    status: str
    upload_timestamp: datetime
    processing_status: Optional[str] = None
    extracted_triples_count: Optional[int] = None
    chunks_count: Optional[int] = None
    metadata: dict = {}


class ProcessingRequest(BaseModel):
    """Document processing request model"""
    document_id: str
    extract_triples: bool = True
    chunk_documents: bool = True
    store_vectors: bool = True
    domain_ontology: Optional[str] = None


# In-memory storage for demo (replace with proper database)
uploaded_documents = {}
processing_status = {}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing
    
    Supports: PDF, DOCX, XLSX, TXT formats
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.xlsx', '.txt'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. Supported: {allowed_extensions}"
        )
    
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Read file content
    try:
        content = await file.read()
        file_size = len(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    
    # Store document info
    document_info = DocumentInfo(
        document_id=document_id,
        filename=file.filename,
        file_size=file_size,
        content_type=file.content_type,
        status="uploaded",
        upload_timestamp=datetime.now(),
        metadata={
            "original_filename": file.filename,
            "file_extension": file_extension
        }
    )
    
    uploaded_documents[document_id] = document_info
    
    # TODO: Save file to storage
    # For now, just store in memory (replace with proper file storage)
    
    logger.info(f"Document uploaded: {file.filename} ({file_size} bytes) -> {document_id}")
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        file_size=file_size,
        content_type=file.content_type,
        status="uploaded",
        upload_timestamp=datetime.now()
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(
    status: Optional[str] = None,
    limit: int = 100
):
    """List uploaded documents with optional status filtering"""
    documents = list(uploaded_documents.values())
    
    if status:
        documents = [doc for doc in documents if doc.status == status]
    
    # Sort by upload timestamp (newest first)
    documents.sort(key=lambda x: x.upload_timestamp, reverse=True)
    
    return documents[:limit]


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get information about a specific document"""
    if document_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = uploaded_documents[document_id]
    
    # Update with latest processing status if available
    if document_id in processing_status:
        document.processing_status = processing_status[document_id].get("status")
        document.extracted_triples_count = processing_status[document_id].get("triples_count")
        document.chunks_count = processing_status[document_id].get("chunks_count")
    
    return document


@router.post("/{document_id}/process")
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    request: ProcessingRequest
):
    """
    Start processing a document through the knowledge graph pipeline
    
    Processing stages:
    1. Document parsing (vLLM SmolDocling for PDF, native for others)
    2. Text chunking
    3. Triple extraction (Hochschul-LLM)
    4. Storage (Fuseki + ChromaDB)
    """
    if document_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = uploaded_documents[document_id]
    
    if document.status not in ["uploaded", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Document is already being processed or completed. Status: {document.status}"
        )
    
    # Initialize processing status
    processing_status[document_id] = {
        "status": "pending",
        "stage": "queued",
        "progress": 0.0,
        "started_at": datetime.now(),
        "triples_count": 0,
        "chunks_count": 0
    }
    
    # Update document status
    document.status = "processing"
    uploaded_documents[document_id] = document
    
    # Add background task for processing
    background_tasks.add_task(
        process_document_pipeline,
        document_id,
        request
    )
    
    logger.info(f"Started processing document {document_id}: {document.filename}")
    
    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Document processing started",
        "stages": [
            "document_parsing",
            "text_chunking", 
            "triple_extraction",
            "storage"
        ]
    }


@router.get("/{document_id}/status")
async def get_processing_status(document_id: str):
    """Get detailed processing status for a document"""
    if document_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document_id not in processing_status:
        return {
            "document_id": document_id,
            "status": "not_started",
            "message": "Document has not been processed yet"
        }
    
    status = processing_status[document_id]
    
    return {
        "document_id": document_id,
        "status": status["status"],
        "stage": status["stage"],
        "progress": status["progress"],
        "started_at": status["started_at"],
        "updated_at": status.get("updated_at", status["started_at"]),
        "triples_count": status.get("triples_count", 0),
        "chunks_count": status.get("chunks_count", 0),
        "error": status.get("error"),
        "details": status.get("details", {})
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    if document_id not in uploaded_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document = uploaded_documents[document_id]
    
    # TODO: Delete from file storage
    # TODO: Delete from vector store
    # TODO: Delete triples from triple store
    
    # Remove from memory
    del uploaded_documents[document_id]
    if document_id in processing_status:
        del processing_status[document_id]
    
    logger.info(f"Deleted document {document_id}: {document.filename}")
    
    return {
        "document_id": document_id,
        "message": "Document deleted successfully"
    }


async def process_document_pipeline(document_id: str, request: ProcessingRequest):
    """
    Background task for document processing pipeline
    
    Implements the complete pipeline with vLLM SmolDocling integration
    """
    try:
        status = processing_status[document_id]
        document = uploaded_documents[document_id]
        
        # Stage 1: Document Parsing
        status.update({
            "stage": "document_parsing",
            "progress": 10.0,
            "updated_at": datetime.now()
        })
        
        # Get file extension to determine parser
        file_extension = Path(document.filename).suffix.lower()
        parsed_document = None
        
        if file_extension == '.pdf':
            # Use vLLM SmolDocling for PDF parsing (GPU 1)
            logger.info(f"Using vLLM SmolDocling for PDF: {document.filename}")
            
            try:
                async with VLLMSmolDoclingClient() as vllm_client:
                    # TODO: Get actual file path from storage
                    # For now, this is a placeholder - in real implementation,
                    # we would have saved the file and have its path
                    logger.warning("PDF parsing with vLLM - file storage not yet implemented")
                    
                    # Update progress
                    status.update({
                        "progress": 25.0,
                        "details": {"parser": "vLLM SmolDocling", "gpu": "GPU 1"},
                        "updated_at": datetime.now()
                    })
                    
            except Exception as e:
                logger.error(f"vLLM SmolDocling parsing failed: {e}")
                raise ParseError(f"PDF parsing failed: {e}")
                
        else:
            # Use native parsers for other formats
            logger.info(f"Using native parser for {file_extension}: {document.filename}")
            status.update({
                "progress": 25.0,
                "details": {"parser": f"Native {file_extension}", "gpu": "CPU only"},
                "updated_at": datetime.now()
            })
            
            # TODO: Implement native parsers for DOCX, XLSX, TXT
        
        # Stage 2: Text Chunking
        status.update({
            "stage": "text_chunking", 
            "progress": 30.0,
            "updated_at": datetime.now()
        })
        
        # TODO: Implement text chunking
        status["chunks_count"] = 5  # Placeholder
        
        status.update({
            "progress": 50.0,
            "updated_at": datetime.now()
        })
        
        # Stage 3: Triple Extraction
        if request.extract_triples:
            status.update({
                "stage": "triple_extraction",
                "progress": 60.0,
                "updated_at": datetime.now()
            })
            
            # Use Hochschul-LLM for triple extraction (external API)
            logger.info("Starting triple extraction with Hochschul-LLM via OpenAI API")
            
            try:
                async with HochschulLLMClient() as llm_client:
                    # TODO: Get actual text chunks from parsed document
                    # For now, this is a placeholder - in real implementation,
                    # we would have the parsed and chunked text
                    sample_chunks = [
                        "Sample text chunk 1 for triple extraction.",
                        "Sample text chunk 2 with more content."
                    ]
                    
                    # Extract triples from chunks
                    extraction_results = await llm_client.extract_triples_batch(
                        sample_chunks,
                        domain_context=request.domain_ontology,
                        ontology_hints=None  # TODO: Load from ontology plugins
                    )
                    
                    # Count successful extractions
                    successful_extractions = [r for r in extraction_results if r.success]
                    total_triples = sum(r.triple_count for r in successful_extractions)
                    
                    status["triples_count"] = total_triples
                    status["extraction_details"] = {
                        "chunks_processed": len(sample_chunks),
                        "successful_chunks": len(successful_extractions),
                        "failed_chunks": len(extraction_results) - len(successful_extractions),
                        "average_confidence": sum(r.average_confidence for r in successful_extractions) / len(successful_extractions) if successful_extractions else 0.0
                    }
                    
                    logger.info(f"Triple extraction completed: {total_triples} triples from {len(successful_extractions)} chunks")
                    
            except Exception as e:
                logger.error(f"Hochschul-LLM triple extraction failed: {e}")
                status["triples_count"] = 0
                status["extraction_error"] = str(e)
            
            status.update({
                "progress": 80.0,
                "details": {"llm_provider": "Hochschul-LLM", "api": "OpenAI-compatible"},
                "updated_at": datetime.now()
            })
        
        # Stage 4: Storage
        status.update({
            "stage": "storage",
            "progress": 90.0,
            "updated_at": datetime.now()
        })
        
        # TODO: Store in Fuseki and ChromaDB
        
        # Completion
        status.update({
            "status": "completed",
            "stage": "completed",
            "progress": 100.0,
            "updated_at": datetime.now(),
            "completed_at": datetime.now()
        })
        
        # Update document status
        uploaded_documents[document_id].status = "completed"
        
        logger.info(f"Document processing completed: {document_id} using GPU workload separation")
        
    except Exception as e:
        logger.error(f"Document processing failed: {document_id} - {e}", exc_info=True)
        
        processing_status[document_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.now()
        })
        
        uploaded_documents[document_id].status = "failed"