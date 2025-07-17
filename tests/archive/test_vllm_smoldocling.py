"""Tests for vLLM SmolDocling client"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.clients.vllm_smoldocling import (
    VLLMSmolDoclingClient,
    SmolDoclingConfig,
    SmolDoclingResult,
    SmolDoclingPage,
    TableData,
    ImageData,
    FormulaData
)
from core.parsers import ParseError


@pytest.fixture
def mock_vllm_response():
    """Mock response from vLLM SmolDocling"""
    return {
        "choices": [{
            "message": {
                "content": '''```json
{
  "pages": [
    {
      "page_number": 1,
      "text": "This is the main text content of page 1.",
      "tables": [
        {
          "caption": "Test Table",
          "headers": ["Column A", "Column B"],
          "rows": [["Value 1", "Value 2"], ["Value 3", "Value 4"]],
          "page_number": 1
        }
      ],
      "images": [
        {
          "caption": "Test Figure",
          "description": "A diagram showing the process flow",
          "page_number": 1,
          "image_type": "diagram"
        }
      ],
      "formulas": [
        {
          "latex": "E = mc^2",
          "description": "Einstein's mass-energy equivalence",
          "page_number": 1
        }
      ]
    }
  ],
  "metadata": {
    "title": "Test Document",
    "author": "Test Author",
    "total_pages": 1
  }
}
```'''
            }
        }],
        "model": "smoldocling-v1.0"
    }


@pytest.fixture
def sample_pdf_file(tmp_path):
    """Create a sample PDF file for testing"""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ntest content\n%%EOF")
    return pdf_path


@pytest.mark.asyncio
class TestVLLMSmolDoclingClient:
    """Test cases for vLLM SmolDocling client"""
    
    async def test_initialization(self):
        """Test client initialization"""
        client = VLLMSmolDoclingClient()
        assert client.endpoint is not None
        assert isinstance(client.config, SmolDoclingConfig)
        
        # Test with custom config
        custom_config = SmolDoclingConfig(max_pages=50, extract_tables=False)
        client_custom = VLLMSmolDoclingClient(custom_config)
        assert client_custom.config.max_pages == 50
        assert client_custom.config.extract_tables is False
    
    async def test_health_check_success(self, mock_vllm_smoldocling):
        """Test successful health check"""
        mock_vllm_smoldocling.get.return_value.json.return_value = {
            "status": "healthy",
            "model_info": {"version": "1.0"},
            "gpu_info": {"memory_usage": "50%"}
        }
        
        client = VLLMSmolDoclingClient()
        health = await client.health_check()
        
        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        assert "model_info" in health
    
    async def test_health_check_failure(self):
        """Test health check failure"""
        client = VLLMSmolDoclingClient()
        
        with patch.object(client.client, 'get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            health = await client.health_check()
            
            assert health["status"] == "unhealthy"
            assert "error" in health
    
    async def test_parse_pdf_success(self, sample_pdf_file, mock_vllm_response):
        """Test successful PDF parsing"""
        client = VLLMSmolDoclingClient()
        
        with patch.object(client.client, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_vllm_response
            mock_response.raise_for_status = AsyncMock()
            mock_post.return_value = mock_response
            
            result = await client.parse_pdf(sample_pdf_file)
            
            assert isinstance(result, SmolDoclingResult)
            assert result.success is True
            assert result.total_pages == 1
            assert len(result.pages) == 1
            
            page = result.pages[0]
            assert page.page_number == 1
            assert page.text == "This is the main text content of page 1."
            assert len(page.tables) == 1
            assert len(page.images) == 1
            assert len(page.formulas) == 1
    
    async def test_parse_pdf_file_not_found(self):
        """Test parsing non-existent file"""
        client = VLLMSmolDoclingClient()
        
        with pytest.raises(ParseError, match="PDF file not found"):
            await client.parse_pdf(Path("nonexistent.pdf"))
    
    async def test_parse_pdf_wrong_extension(self, tmp_path):
        """Test parsing file with wrong extension"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        
        client = VLLMSmolDoclingClient()
        
        with pytest.raises(ParseError, match="Expected PDF file"):
            await client.parse_pdf(txt_file)
    
    async def test_parse_pdf_http_error(self, sample_pdf_file):
        """Test HTTP error during parsing"""
        client = VLLMSmolDoclingClient()
        
        with patch.object(client.client, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response
            mock_response.raise_for_status.side_effect = Exception("HTTP 500")
            
            with pytest.raises(ParseError, match="PDF parsing failed"):
                await client.parse_pdf(sample_pdf_file)
    
    async def test_convert_to_document(self, sample_pdf_file, mock_vllm_response):
        """Test conversion to Document format"""
        client = VLLMSmolDoclingClient()
        
        # Create a SmolDocling result manually
        table = TableData(
            caption="Test Table",
            headers=["Column A", "Column B"],
            rows=[["Value 1", "Value 2"]],
            page_number=1
        )
        
        image = ImageData(
            caption="Test Figure",
            description="A test image",
            page_number=1,
            image_type="diagram"
        )
        
        formula = FormulaData(
            latex="E = mc^2",
            description="Einstein's equation",
            page_number=1
        )
        
        page = SmolDoclingPage(
            page_number=1,
            text="Test content",
            tables=[table],
            images=[image],
            formulas=[formula],
            layout_info={}
        )
        
        result = SmolDoclingResult(
            pages=[page],
            metadata={"title": "Test Document"},
            processing_time_seconds=1.5,
            model_version="smoldocling-v1.0",
            total_pages=1,
            success=True
        )
        
        document = client.convert_to_document(result, sample_pdf_file)
        
        assert document.content is not None
        assert len(document.segments) == 4  # text + table + image + formula
        assert document.metadata.title == "Test Document"
        assert document.metadata.page_count == 1
        
        # Check segment types
        segment_types = [seg.segment_type for seg in document.segments]
        assert "text" in segment_types
        assert "table" in segment_types
        assert "image_caption" in segment_types
        assert "formula" in segment_types
    
    async def test_convert_failed_result(self, sample_pdf_file):
        """Test conversion of failed parsing result"""
        client = VLLMSmolDoclingClient()
        
        failed_result = SmolDoclingResult(
            pages=[],
            metadata={},
            processing_time_seconds=0.0,
            model_version="unknown",
            total_pages=0,
            success=False,
            error_message="Parsing failed"
        )
        
        with pytest.raises(ParseError, match="Cannot convert failed parsing result"):
            client.convert_to_document(failed_result, sample_pdf_file)
    
    def test_table_to_text(self):
        """Test table to text conversion"""
        client = VLLMSmolDoclingClient()
        
        table = TableData(
            caption="Test Table",
            headers=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
            page_number=1
        )
        
        text = client._table_to_text(table)
        
        assert "Test Table" in text
        assert "A | B" in text
        assert "1 | 2" in text
        assert "3 | 4" in text
    
    def test_build_parsing_prompt(self):
        """Test parsing prompt generation"""
        client = VLLMSmolDoclingClient()
        config = SmolDoclingConfig(
            extract_tables=True,
            extract_images=True,
            extract_formulas=False,
            max_pages=5
        )
        
        prompt = client._build_parsing_prompt(config)
        
        assert "JSON format" in prompt
        assert "Extract all tables" in prompt
        assert "Analyze images" in prompt
        assert "Process up to 5 pages" in prompt
        assert "mathematical formulas" not in prompt  # Should not be included
    
    async def test_batch_parse_pdfs(self, tmp_path, mock_vllm_response):
        """Test batch PDF parsing"""
        # Create multiple PDF files
        pdf_files = []
        for i in range(3):
            pdf_path = tmp_path / f"test_{i}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\ntest content\n%%EOF")
            pdf_files.append(pdf_path)
        
        client = VLLMSmolDoclingClient()
        
        with patch.object(client, 'parse_pdf') as mock_parse:
            # Mock successful parsing for all files
            mock_parse.return_value = SmolDoclingResult(
                pages=[],
                metadata={},
                processing_time_seconds=1.0,
                model_version="test",
                total_pages=1,
                success=True
            )
            
            results = await client.batch_parse_pdfs(pdf_files)
            
            assert len(results) == 3
            assert all(result.success for result in results)
            assert mock_parse.call_count == 3
    
    async def test_context_manager(self):
        """Test async context manager usage"""
        async with VLLMSmolDoclingClient() as client:
            assert client is not None
            assert hasattr(client, 'client')
        
        # Client should be closed after context manager exit