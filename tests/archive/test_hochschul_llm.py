"""Tests for Hochschul-LLM client"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.clients.hochschul_llm import (
    HochschulLLMClient,
    TripleExtractionConfig,
    Triple,
    ExtractionResult
)


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI API"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps({
        "triples": [
            {
                "subject": "John Doe",
                "predicate": "worksAt",
                "object": "Example Corp",
                "confidence": 0.95,
                "context": "John Doe is an employee at Example Corp"
            },
            {
                "subject": "Example Corp",
                "predicate": "locatedIn",
                "object": "New York",
                "confidence": 0.90,
                "context": "Example Corp is based in New York"
            }
        ],
        "metadata": {
            "extraction_approach": "Named entity recognition and relationship extraction",
            "domain_detected": "business",
            "quality_indicators": ["clear entities", "explicit relationships"]
        }
    })
    response.model = "qwen1.5-72b"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 150
    response.usage.completion_tokens = 80
    response.usage.total_tokens = 230
    response.usage.model_dump.return_value = {
        "prompt_tokens": 150,
        "completion_tokens": 80,
        "total_tokens": 230
    }
    return response


@pytest.fixture
def sample_text():
    """Sample text for triple extraction"""
    return """
    John Doe is a software engineer at Example Corp. The company is headquartered 
    in New York and specializes in artificial intelligence solutions. John has been 
    working there since 2020 and leads the machine learning team.
    """


@pytest.mark.asyncio
class TestHochschulLLMClient:
    """Test cases for Hochschul-LLM client"""
    
    def test_initialization_with_config(self, test_config):
        """Test client initialization with configuration"""
        custom_config = TripleExtractionConfig(
            temperature=0.2,
            max_tokens=2000,
            batch_size=3
        )
        
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient(custom_config)
            
            assert client.config.temperature == 0.2
            assert client.config.max_tokens == 2000
            assert client.config.batch_size == 3
            assert client.endpoint == test_config.llm.hochschul.endpoint
            assert client.model == test_config.llm.hochschul.model
    
    def test_initialization_without_config(self):
        """Test client initialization failure without config"""
        with patch("core.clients.hochschul_llm.get_config") as mock_config:
            mock_config.return_value.llm.hochschul = None
            
            with pytest.raises(ValueError, match="Hochschul-LLM configuration not found"):
                HochschulLLMClient()
    
    async def test_health_check_success(self, test_config, mock_openai_response):
        """Test successful health check"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.return_value = mock_openai_response
                mock_openai_response.choices[0].message.content = "OK"
                
                health = await client.health_check()
                
                assert health["status"] == "healthy"
                assert health["endpoint"] == test_config.llm.hochschul.endpoint
                assert health["model"] == test_config.llm.hochschul.model
                assert "response_time_ms" in health
                assert health["test_response"] == "OK"
    
    async def test_health_check_failure(self, test_config):
        """Test health check failure"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.side_effect = Exception("Connection failed")
                
                health = await client.health_check()
                
                assert health["status"] == "unhealthy"
                assert "error" in health
                assert "Connection failed" in health["error"]
    
    async def test_extract_triples_success(self, test_config, sample_text, mock_openai_response):
        """Test successful triple extraction"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.return_value = mock_openai_response
                
                result = await client.extract_triples(sample_text)
                
                assert isinstance(result, ExtractionResult)
                assert result.success is True
                assert result.triple_count == 2
                assert len(result.triples) == 2
                
                # Check first triple
                triple1 = result.triples[0]
                assert triple1.subject == "John Doe"
                assert triple1.predicate == "worksAt"
                assert triple1.object == "Example Corp"
                assert triple1.confidence == 0.95
                
                # Check metadata
                assert "extraction_approach" in result.metadata
                assert result.confidence_scores["overall"] > 0.9
    
    async def test_extract_triples_with_context(self, test_config, sample_text, mock_openai_response):
        """Test triple extraction with domain context and ontology hints"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.return_value = mock_openai_response
                
                result = await client.extract_triples(
                    sample_text,
                    domain_context="business and employment",
                    ontology_hints=["worksAt", "hasRole", "locatedIn"]
                )
                
                assert result.success is True
                
                # Check that the prompt was called with domain context
                call_args = mock_chat.completions.create.call_args
                prompt_content = call_args[1]["messages"][1]["content"]
                assert "business and employment" in prompt_content
                assert "worksAt" in prompt_content
    
    async def test_extract_triples_json_parsing_error(self, test_config, sample_text):
        """Test handling of JSON parsing errors"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                # Mock response with invalid JSON
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "Invalid JSON response"
                mock_response.model = "qwen1.5-72b"
                mock_response.usage = None
                
                mock_chat.completions.create.return_value = mock_response
                
                result = await client.extract_triples(sample_text)
                
                assert result.success is False
                assert "Response parsing failed" in result.error_message
                assert result.triple_count == 0
    
    async def test_extract_triples_api_error(self, test_config, sample_text):
        """Test handling of API errors"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.side_effect = Exception("API Error")
                
                result = await client.extract_triples(sample_text)
                
                assert result.success is False
                assert "API Error" in result.error_message
                assert result.triple_count == 0
    
    async def test_extract_triples_batch(self, test_config, mock_openai_response):
        """Test batch triple extraction"""
        text_chunks = [
            "John works at Example Corp.",
            "Example Corp is in New York.",
            "The company was founded in 2010."
        ]
        
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client.client, 'chat') as mock_chat:
                mock_chat.completions.create.return_value = mock_openai_response
                
                results = await client.extract_triples_batch(text_chunks)
                
                assert len(results) == 3
                assert all(isinstance(r, ExtractionResult) for r in results)
                assert all(r.success for r in results)
                
                # Check that API was called for each chunk
                assert mock_chat.completions.create.call_count == 3
    
    async def test_extract_triples_batch_with_error(self, test_config):
        """Test batch processing with some failures"""
        text_chunks = ["Text 1", "Text 2", "Text 3"]
        
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            with patch.object(client, 'extract_triples') as mock_extract:
                # Mock first success, second failure, third success
                mock_extract.side_effect = [
                    ExtractionResult([], "Text 1", 1.0, "model", {}, {}, True),
                    Exception("API Error"),
                    ExtractionResult([], "Text 3", 1.0, "model", {}, {}, True)
                ]
                
                results = await client.extract_triples_batch(text_chunks)
                
                assert len(results) == 3
                assert results[0].success is True
                assert results[1].success is False  # Should be failed result
                assert results[2].success is True
    
    def test_build_triple_extraction_prompt(self, test_config):
        """Test prompt building with various options"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            # Test basic prompt
            prompt = client._build_triple_extraction_prompt("Test text")
            assert "Extract RDF triples" in prompt
            assert "Test text" in prompt
            assert "JSON format" in prompt
            
            # Test with domain context
            prompt_with_domain = client._build_triple_extraction_prompt(
                "Test text",
                domain_context="scientific research"
            )
            assert "scientific research" in prompt_with_domain
            
            # Test with ontology hints
            prompt_with_ontology = client._build_triple_extraction_prompt(
                "Test text",
                ontology_hints=["hasAuthor", "publishedIn"]
            )
            assert "hasAuthor" in prompt_with_ontology
            assert "publishedIn" in prompt_with_ontology
    
    def test_triple_data_structures(self):
        """Test Triple and ExtractionResult data structures"""
        # Test Triple
        triple = Triple(
            subject="Subject",
            predicate="predicate",
            object="Object",
            confidence=0.95
        )
        
        assert triple.subject == "Subject"
        assert triple.confidence == 0.95
        assert triple.metadata == {}  # Should be initialized
        
        # Test to_dict method
        triple_dict = triple.to_dict()
        assert triple_dict["subject"] == "Subject"
        assert triple_dict["confidence"] == 0.95
        
        # Test to_ntriples method
        ntriples = triple.to_ntriples()
        assert "<Subject>" in ntriples
        assert "<predicate>" in ntriples
        assert '"Object"' in ntriples
        
        # Test ExtractionResult
        result = ExtractionResult(
            triples=[triple],
            source_text="Test text",
            processing_time_seconds=1.5,
            model_used="test-model",
            confidence_scores={"overall": 0.95},
            metadata={},
            success=True
        )
        
        assert result.triple_count == 1
        assert result.average_confidence == 0.95
        
        # Test confidence filtering
        filtered = result.filter_by_confidence(0.9)
        assert len(filtered) == 1
        
        filtered_high = result.filter_by_confidence(0.99)
        assert len(filtered_high) == 0
    
    async def test_validate_triples(self, test_config):
        """Test triple validation functionality"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            client = HochschulLLMClient()
            
            triples = [
                Triple("John Doe", "worksAt", "Example Corp", 0.95),
                Triple("", "hasAge", "30", 0.80),  # Empty subject
                Triple("Example Corp", "locatedIn", "New York", 0.60),  # Low confidence
                Triple("Something", "relatedTo", "thing", 0.85)  # Generic terms
            ]
            
            validation = await client.validate_triples(triples)
            
            assert validation["total_triples"] == 4
            assert validation["valid_triples"] == 1  # Only first triple is fully valid
            assert validation["quality_score"] == 0.25
            assert len(validation["issues"]) == 3
            assert len(validation["recommendations"]) > 0
    
    async def test_context_manager(self, test_config):
        """Test async context manager usage"""
        with patch("core.clients.hochschul_llm.get_config", return_value=test_config):
            async with HochschulLLMClient() as client:
                assert client is not None
                assert hasattr(client, 'client')
        
        # Client should be closed after context manager exit
    
    def test_triple_extraction_config(self):
        """Test TripleExtractionConfig model"""
        # Test default values
        config = TripleExtractionConfig()
        assert config.model == "qwen1.5-72b"
        assert config.temperature == 0.1
        assert config.max_tokens == 4000
        assert config.batch_size == 5
        
        # Test custom values
        custom_config = TripleExtractionConfig(
            temperature=0.2,
            max_tokens=2000,
            confidence_threshold=0.8
        )
        assert custom_config.temperature == 0.2
        assert custom_config.max_tokens == 2000
        assert custom_config.confidence_threshold == 0.8