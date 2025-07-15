"""
Hochschul-LLM client for triple extraction via OpenAI-compatible API
Modernized with standardized BaseModelClient architecture
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.clients.base import BaseModelClient, BatchProcessingMixin
from core.config_new.unified_manager import get_config

logger = logging.getLogger(__name__)


# Data Models
class TripleExtractionConfig(BaseModel):
    """Configuration for triple extraction"""
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    chunk_overlap_strategy: str = "context_aware"
    confidence_threshold: float = 0.7
    batch_size: int = 5


class Triple(BaseModel):
    """Extracted RDF triple"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    source_chunk: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert triple to dictionary format"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_chunk": self.source_chunk,
            "context": self.context,
            "metadata": self.metadata
        }
    
    def to_ntriples(self) -> str:
        """Convert triple to N-Triples format"""
        return f'<{self.subject}> <{self.predicate}> "{self.object}" .'


class ExtractionResult(BaseModel):
    """Triple extraction result from Hochschul-LLM"""
    triples: List[Triple]
    source_text: str
    processing_time_seconds: float
    model_used: str
    confidence_scores: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}
    success: bool
    error_message: Optional[str] = None
    
    @property
    def triple_count(self) -> int:
        """Number of extracted triples"""
        return len(self.triples)
    
    @property
    def average_confidence(self) -> float:
        """Average confidence score across all triples"""
        if not self.triples:
            return 0.0
        return sum(t.confidence for t in self.triples) / len(self.triples)
    
    def filter_by_confidence(self, threshold: float) -> List[Triple]:
        """Filter triples by confidence threshold"""
        return [t for t in self.triples if t.confidence >= threshold]


class ExtractionRequest(BaseModel):
    """Request for triple extraction"""
    text: str
    prompt_template: Optional[str] = None
    ontology_context: Optional[str] = None
    max_triples: int = 50
    language: str = "de"


class HochschulLLMClient(BaseModelClient[ExtractionRequest, ExtractionResult, TripleExtractionConfig],
                        BatchProcessingMixin):
    """
    Modernized client for Hochschul-LLM triple extraction
    
    Benefits over original:
    - Automatic retry logic for API failures
    - Standardized health checks
    - Built-in metrics collection
    - Batch processing support
    - Unified error handling
    """
    
    def __init__(self, config: Optional[TripleExtractionConfig] = None):
        """Initialize with backward compatibility"""
        # OpenAI client needs special handling
        self._openai_client = None
        super().__init__("hochschul_llm", config=config)
        
    def _get_default_config(self) -> TripleExtractionConfig:
        """Default configuration for triple extraction"""
        return TripleExtractionConfig()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client if not already done"""
        if self._openai_client is None:
            system_config = get_config()
            self._openai_client = AsyncOpenAI(
                api_key=system_config.services.hochschul_llm.api_key,
                base_url=self.endpoint,
                timeout=self.timeout,
                max_retries=0  # We handle retries in BaseModelClient
            )
            self.model = system_config.services.hochschul_llm.model
    
    async def close(self):
        """Close connections"""
        if self._openai_client:
            await self._openai_client.close()
        await super().close()
    
    async def _process_internal(self, request: ExtractionRequest) -> ExtractionResult:
        """
        Internal triple extraction processing
        
        Args:
            request: Text and extraction parameters
            
        Returns:
            Extraction result with triples
        """
        self._initialize_openai_client()
        start_time = datetime.now()
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(
            request.text,
            request.prompt_template,
            request.ontology_context,
            request.max_triples,
            request.language
        )
        
        try:
            # Call LLM
            response = await self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a knowledge graph extraction expert. Extract RDF triples from the given text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_json = json.loads(response.choices[0].message.content)
            
            # Convert to Triple objects
            triples = []
            for triple_data in result_json.get("triples", []):
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    confidence=triple_data.get("confidence", 0.8),
                    source_chunk=request.text[:200],  # First 200 chars as context
                    context=triple_data.get("context", "")
                )
                triples.append(triple)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ExtractionResult(
                triples=triples,
                source_text=request.text,
                processing_time_seconds=processing_time,
                model_used=self.model,
                confidence_scores={
                    "average": sum(t.confidence for t in triples) / len(triples) if triples else 0,
                    "min": min(t.confidence for t in triples) if triples else 0,
                    "max": max(t.confidence for t in triples) if triples else 0
                },
                metadata={
                    "language": request.language,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                success=True
            )
            
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return ExtractionResult(
                triples=[],
                source_text=request.text,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                model_used=self.model,
                success=False,
                error_message=str(e)
            )
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Hochschul-LLM specific health check"""
        self._initialize_openai_client()
        
        # Test with simple completion
        response = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Test. Reply with OK."}],
            max_tokens=10,
            temperature=0
        )
        
        return {
            "model_available": True,
            "model_name": self.model,
            "response": response.choices[0].message.content,
            "api_version": "openai-compatible"
        }
    
    def _build_extraction_prompt(self, 
                                text: str, 
                                template: Optional[str],
                                ontology: Optional[str],
                                max_triples: int,
                                language: str) -> str:
        """Build the extraction prompt"""
        if template:
            return template.format(text=text, max_triples=max_triples)
        
        base_prompt = f"""Extract knowledge graph triples from the following text.
Return the result as JSON with the following structure:
{{
    "triples": [
        {{
            "subject": "Subject entity",
            "predicate": "Relationship/Property",
            "object": "Object entity or value",
            "confidence": 0.0-1.0,
            "context": "Optional context"
        }}
    ]
}}

Rules:
- Extract up to {max_triples} triples
- Language: {language}
- Focus on factual relationships
- Include confidence scores
- Preserve original entity names
"""
        
        if ontology:
            base_prompt += f"\n\nUse this ontology context:\n{ontology}\n"
        
        base_prompt += f"\n\nText to analyze:\n{text}"
        
        return base_prompt
    
    # Convenience methods for backward compatibility
    async def extract_triples(self, 
                            text: str,
                            prompt_template: Optional[str] = None,
                            ontology_context: Optional[str] = None) -> ExtractionResult:
        """
        Extract triples from text (backward compatibility)
        
        Args:
            text: Source text
            prompt_template: Custom prompt template
            ontology_context: Ontology information
            
        Returns:
            Extraction result
        """
        request = ExtractionRequest(
            text=text,
            prompt_template=prompt_template,
            ontology_context=ontology_context
        )
        return await self.process(request)
    
    async def extract_from_chunks(self, 
                                chunks: List[str],
                                batch_size: Optional[int] = None) -> List[ExtractionResult]:
        """
        Extract triples from multiple text chunks
        
        Args:
            chunks: List of text chunks
            batch_size: Override default batch size
            
        Returns:
            List of extraction results
        """
        requests = [ExtractionRequest(text=chunk) for chunk in chunks]
        
        return await self.process_batch(
            requests,
            batch_size=batch_size or self.config.batch_size,
            concurrent_batches=3
        )


# Example usage
async def example_usage():
    """Show benefits of new architecture"""
    
    async with HochschulLLMClient() as client:
        # 1. Health check
        health = await client.health_check()
        print(f"Service Status: {health.status}")
        
        # 2. Single extraction with auto-retry
        result = await client.extract_triples(
            "Die Universität Heidelberg wurde 1386 gegründet."
        )
        print(f"Extracted {result.triple_count} triples")
        
        # 3. Batch extraction
        chunks = [
            "Berlin ist die Hauptstadt von Deutschland.",
            "Die Elbe fließt durch Hamburg.",
            "München liegt in Bayern."
        ]
        results = await client.extract_from_chunks(chunks)
        print(f"Processed {len(results)} chunks")
        
        # 4. Get metrics
        metrics = client.get_metrics()
        print(f"Average response time: {metrics.average_response_time_ms}ms")