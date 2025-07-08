"""Pytest configuration and fixtures"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.config import Config, HochschulLLMConfig, LLMConfig


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Config:
    """Create test configuration"""
    config = Config()
    
    # Configure test Hochschul LLM
    config.llm = LLMConfig(
        provider="hochschul",
        hochschul=HochschulLLMConfig(
            endpoint="https://test.hochschule.example/api",
            api_key="test-api-key",
            model="qwen1.5-72b",
            temperature=0.1,
            max_tokens=4000,
            timeout=60
        )
    )
    
    # Override with test endpoints
    config.parsing.pdf.vllm_endpoint = "http://localhost:8002"
    config.storage.triple_store.endpoint = "http://localhost:3030"
    config.storage.vector_store.endpoint = "http://localhost:8001"
    
    return config


@pytest.fixture
def mock_hochschul_llm():
    """Mock for Hochschul LLM API calls"""
    with patch("httpx.AsyncClient") as mock_client:
        # Create mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "triples": [
                            {
                                "subject": "Test Document",
                                "predicate": "contains",
                                "object": "Information"
                            },
                            {
                                "subject": "Information",
                                "predicate": "type",
                                "object": "TestData"
                            }
                        ]
                    })
                }
            }]
        }
        
        # Configure mock client
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_vllm_smoldocling():
    """Mock for vLLM SmolDocling PDF parsing"""
    with patch("httpx.AsyncClient") as mock_client:
        # Create mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "pages": [
                            {
                                "page_number": 1,
                                "text": "This is page 1 content.",
                                "tables": [],
                                "images": []
                            },
                            {
                                "page_number": 2,
                                "text": "This is page 2 content.",
                                "tables": [
                                    {
                                        "caption": "Test Table",
                                        "data": [["A", "B"], ["1", "2"]]
                                    }
                                ],
                                "images": []
                            }
                        ],
                        "metadata": {
                            "total_pages": 2,
                            "title": "Test Document",
                            "author": "Test Author"
                        }
                    })
                }
            }]
        }
        
        # Configure mock client
        mock_instance = AsyncMock()
        mock_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_fuseki():
    """Mock for Fuseki triple store"""
    with patch("SPARQLWrapper.SPARQLWrapper") as mock_sparql:
        mock_instance = MagicMock()
        
        # Mock query results
        mock_results = {
            "results": {
                "bindings": [
                    {
                        "s": {"value": "http://example.org/subject1"},
                        "p": {"value": "http://example.org/predicate1"},
                        "o": {"value": "Object1"}
                    }
                ]
            }
        }
        
        mock_instance.query().convert.return_value = mock_results
        mock_sparql.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_chromadb():
    """Mock for ChromaDB vector store"""
    with patch("chromadb.Client") as mock_client:
        mock_collection = MagicMock()
        
        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["doc1_chunk1", "doc1_chunk2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["Chunk 1 content", "Chunk 2 content"]],
            "metadatas": [[{"page": 1}, {"page": 2}]]
        }
        
        mock_instance = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_instance
        
        yield mock_collection


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a sample PDF file for testing"""
    pdf_path = tmp_path / "test_document.pdf"
    # Create a minimal PDF content (simplified for testing)
    pdf_path.write_bytes(b"%PDF-1.4\ntest content\n%%EOF")
    return pdf_path


@pytest.fixture
def sample_docx_path(tmp_path: Path) -> Path:
    """Create a sample DOCX file for testing"""
    docx_path = tmp_path / "test_document.docx"
    # Create a minimal DOCX content (simplified for testing)
    docx_path.write_bytes(b"PK\x03\x04test content")
    return docx_path


@pytest.fixture
def sample_txt_path(tmp_path: Path) -> Path:
    """Create a sample text file for testing"""
    txt_path = tmp_path / "test_document.txt"
    txt_path.write_text("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
    return txt_path


@pytest.fixture
def api_client(test_config: Config) -> Generator[TestClient, None, None]:
    """Create FastAPI test client"""
    # Import here to avoid circular imports
    from api.main import app
    
    # Override config for testing
    with patch("core.config.get_config", return_value=test_config):
        with TestClient(app) as client:
            yield client


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset config singleton
    import core.config
    core.config._config = None
    
    yield
    
    # Cleanup after test
    core.config._config = None


@pytest.fixture
def env_vars(monkeypatch):
    """Set test environment variables"""
    test_env = {
        "HOCHSCHUL_LLM_ENDPOINT": "https://test.hochschule.example/api",
        "HOCHSCHUL_LLM_API_KEY": "test-api-key",
        "VLLM_SMOLDOCLING_URL": "http://localhost:8002",
        "FUSEKI_URL": "http://localhost:3030",
        "CHROMADB_URL": "http://localhost:8001",
        "OLLAMA_URL": "http://localhost:11434"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env