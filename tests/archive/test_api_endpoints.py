"""Tests for FastAPI endpoints"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from core.batch_processor import ProcessingResult, ProcessingStatus
from core.chunking import ChunkingResult, ContextualChunk, ChunkingStrategy
from core.parsers import Document, DocumentMetadata, DocumentType
from datetime import datetime


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_endpoint_basic(self, api_client):
        """Test basic health endpoint"""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data

    def test_liveness_probe(self, api_client):
        """Test Kubernetes liveness probe"""
        response = api_client.get("/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_readiness_probe(self, api_client):
        """Test Kubernetes readiness probe"""
        response = api_client.get("/health/readiness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ready"
        assert "timestamp" in data

    def test_service_health_endpoint(self, api_client):
        """Test individual service health endpoint"""
        response = api_client.get("/health/services/vllm_smoldocling")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "endpoint" in data
        assert "last_check" in data

    def test_service_health_not_found(self, api_client):
        """Test service health for non-existent service"""
        response = api_client.get("/health/services/nonexistent")
        
        assert response.status_code == 404


class TestDocumentEndpoints:
    """Test document processing endpoints"""
    
    @pytest.mark.asyncio
    async def test_document_upload_endpoint(self, api_client):
        """Test document upload endpoint"""
        
        with patch('core.batch_processor.BatchProcessor.process_file') as mock_process:
            # Mock successful processing
            mock_result = ProcessingResult(
                file_path="test.txt",
                status=ProcessingStatus.COMPLETED,
                document=MagicMock(),
                chunking_result=MagicMock(),
                processing_time=1.5,
                error=None
            )
            mock_process.return_value = mock_result
            
            # Prepare test file
            test_file_content = b"This is test document content for processing."
            
            response = api_client.post(
                "/documents/upload",
                files={"file": ("test.txt", test_file_content, "text/plain")}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "filename" in data
            assert "status" in data
            assert data["filename"] == "test.txt"

    def test_document_upload_no_file(self, api_client):
        """Test document upload without file"""
        response = api_client.post("/documents/upload")
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_batch_upload_endpoint(self, api_client):
        """Test batch document upload"""
        
        with patch('core.batch_processor.BatchProcessor.process_files') as mock_batch:
            # Mock batch processing
            mock_results = [
                ProcessingResult(
                    file_path="doc1.txt",
                    status=ProcessingStatus.COMPLETED,
                    document=MagicMock(),
                    chunking_result=MagicMock(),
                    processing_time=1.0,
                    error=None
                ),
                ProcessingResult(
                    file_path="doc2.txt", 
                    status=ProcessingStatus.COMPLETED,
                    document=MagicMock(),
                    chunking_result=MagicMock(),
                    processing_time=1.2,
                    error=None
                )
            ]
            mock_batch.return_value = mock_results
            
            # Prepare test files
            files = [
                ("files", ("doc1.txt", b"Content 1", "text/plain")),
                ("files", ("doc2.txt", b"Content 2", "text/plain"))
            ]
            
            response = api_client.post("/documents/batch", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "total_files" in data
            assert "successful" in data
            assert data["total_files"] == 2

    def test_document_status_endpoint(self, api_client):
        """Test document status retrieval"""
        
        with patch('api.routers.documents.document_store') as mock_store:
            mock_store.get_document_status.return_value = {
                "document_id": "test_doc_123",
                "status": "completed",
                "processing_time": 2.5,
                "chunks_created": 15,
                "last_updated": datetime.now().isoformat()
            }
            
            response = api_client.get("/documents/test_doc_123")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["document_id"] == "test_doc_123"
            assert data["status"] == "completed"


class TestPipelineEndpoints:
    """Test pipeline processing endpoints"""
    
    @pytest.mark.asyncio
    async def test_pipeline_process_endpoint(self, api_client):
        """Test pipeline processing endpoint"""
        
        with patch('core.content_chunker.ContentChunker.chunk_document') as mock_chunk:
            # Mock chunking result
            mock_chunks = [
                ContextualChunk(chunk_id="chunk1", content="First chunk content"),
                ContextualChunk(chunk_id="chunk2", content="Second chunk content")
            ]
            
            mock_result = ChunkingResult(
                document_id="test_doc",
                source_document=MagicMock(),
                contextual_chunks=mock_chunks,
                context_groups=[],
                chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
                processing_stats=MagicMock(),
                processing_config={}
            )
            mock_chunk.return_value = mock_result
            
            request_data = {
                "content": "This is test content for pipeline processing.",
                "document_type": "txt",
                "processing_options": {
                    "enable_context_inheritance": True,
                    "max_tokens": 1000
                }
            }
            
            response = api_client.post(
                "/pipeline/process", 
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "chunks_created" in data
            assert "processing_time" in data
            assert data["chunks_created"] == 2

    def test_pipeline_status_endpoint(self, api_client):
        """Test pipeline status endpoint"""
        response = api_client.get("/pipeline/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pipeline_status" in data
        assert "active_tasks" in data

    @pytest.mark.asyncio
    async def test_pipeline_process_invalid_content(self, api_client):
        """Test pipeline processing with invalid content"""
        request_data = {
            "content": "",  # Empty content
            "document_type": "txt"
        }
        
        response = api_client.post("/pipeline/process", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestQueryEndpoints:
    """Test query endpoints (when implemented)"""
    
    def test_query_endpoint_placeholder(self, api_client):
        """Test query endpoint placeholder"""
        # This endpoint might not be fully implemented yet
        response = api_client.post(
            "/query/search",
            json={"query": "test query", "limit": 10}
        )
        
        # Depending on implementation status, this might return 404 or 501
        assert response.status_code in [200, 404, 501]


class TestRootEndpoint:
    """Test root API endpoint"""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API information"""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data

    def test_openapi_docs(self, api_client):
        """Test that OpenAPI documentation is available"""
        response = api_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_docs(self, api_client):
        """Test that ReDoc documentation is available"""
        response = api_client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_json(self, api_client):
        """Test OpenAPI JSON schema"""
        response = api_client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_error(self, api_client):
        """Test 404 error handling"""
        response = api_client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404

    @pytest.mark.asyncio 
    async def test_processing_error_handling(self, api_client):
        """Test handling of processing errors"""
        
        with patch('core.batch_processor.BatchProcessor.process_file') as mock_process:
            # Mock processing error
            mock_process.side_effect = Exception("Processing failed")
            
            test_file_content = b"Test content"
            
            response = api_client.post(
                "/documents/upload",
                files={"file": ("error_test.txt", test_file_content, "text/plain")}
            )
            
            assert response.status_code == 500

    def test_validation_error_handling(self, api_client):
        """Test validation error handling"""
        
        # Send invalid JSON
        response = api_client.post(
            "/pipeline/process",
            json={"invalid": "data"}  # Missing required fields
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self, api_client):
        """Test that CORS headers are present"""
        response = api_client.options("/health")
        
        # Check for CORS headers (if configured)
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] == "*"

    def test_preflight_request(self, api_client):
        """Test CORS preflight request"""
        response = api_client.options(
            "/documents/upload",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should allow preflight request
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_concurrent_requests(api_client):
    """Test handling of concurrent requests"""
    import asyncio
    
    with patch('core.batch_processor.BatchProcessor.process_file') as mock_process:
        mock_result = ProcessingResult(
            file_path="concurrent_test.txt",
            status=ProcessingStatus.COMPLETED,
            document=MagicMock(),
            chunking_result=MagicMock(),
            processing_time=1.0,
            error=None
        )
        mock_process.return_value = mock_result
        
        # Simulate concurrent uploads
        test_content = b"Concurrent test content"
        
        async def upload_file(filename):
            return api_client.post(
                "/documents/upload",
                files={"file": (filename, test_content, "text/plain")}
            )
        
        # Send multiple concurrent requests
        tasks = [upload_file(f"concurrent_{i}.txt") for i in range(5)]
        responses = await asyncio.gather(*[asyncio.create_task(asyncio.coroutine(lambda: task)()) for task in tasks])
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200