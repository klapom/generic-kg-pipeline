"""Hochschul-LLM client for triple extraction via OpenAI-compatible API"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI
from pydantic import BaseModel

from core.config import get_config

logger = logging.getLogger(__name__)


class TripleExtractionConfig(BaseModel):
    """Configuration for triple extraction"""
    model: str = "qwen1.5-72b"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    chunk_overlap_strategy: str = "context_aware"  # none, simple, context_aware
    confidence_threshold: float = 0.7
    batch_size: int = 5


@dataclass
class Triple:
    """Extracted RDF triple"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    source_chunk: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
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
        # Simple N-Triples formatting (would need proper URI encoding in production)
        return f'<{self.subject}> <{self.predicate}> "{self.object}" .'


@dataclass
class ExtractionResult:
    """Triple extraction result from Hochschul-LLM"""
    triples: List[Triple]
    source_text: str
    processing_time_seconds: float
    model_used: str
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
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


class HochschulLLMClient:
    """
    Client for Hochschul-LLM triple extraction via OpenAI-compatible API
    
    This client handles knowledge graph triple extraction from text chunks
    using the external Hochschul-LLM infrastructure with Qwen1.5-based models.
    """
    
    def __init__(self, config: Optional[TripleExtractionConfig] = None):
        """Initialize the Hochschul-LLM client"""
        self.config = config or TripleExtractionConfig()
        
        # Get configuration
        system_config = get_config()
        
        if not system_config.llm.hochschul:
            raise ValueError("Hochschul-LLM configuration not found. Please set HOCHSCHUL_LLM_ENDPOINT and HOCHSCHUL_LLM_API_KEY")
        
        # Initialize OpenAI client with Hochschul-LLM endpoint
        self.client = AsyncOpenAI(
            api_key=system_config.llm.hochschul.api_key,
            base_url=system_config.llm.hochschul.endpoint,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries
        )
        
        self.endpoint = system_config.llm.hochschul.endpoint
        self.model = system_config.llm.hochschul.model
        
        logger.info(f"Initialized Hochschul-LLM client: {self.endpoint}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the Hochschul-LLM service is healthy
        
        Returns:
            Health status information
        """
        try:
            start_time = datetime.now()
            
            # Test with a simple completion request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Test connection. Respond with 'OK'."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check if we got a valid response
            if response.choices and response.choices[0].message.content:
                return {
                    "status": "healthy",
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "response_time_ms": response_time,
                    "test_response": response.choices[0].message.content.strip(),
                    "model_info": {
                        "id": response.model,
                        "usage": response.usage.model_dump() if response.usage else {}
                    },
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "endpoint": self.endpoint,
                    "error": "Empty response from model",
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Hochschul-LLM health check failed: {e}")
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def extract_triples(
        self,
        text: str,
        domain_context: Optional[str] = None,
        ontology_hints: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Extract RDF triples from text using Hochschul-LLM
        
        Args:
            text: Input text for triple extraction
            domain_context: Optional domain-specific context
            ontology_hints: Optional ontology relationships to focus on
            
        Returns:
            ExtractionResult with extracted triples
        """
        start_time = datetime.now()
        
        try:
            # Build extraction prompt
            prompt = self._build_triple_extraction_prompt(
                text, domain_context, ontology_hints
            )
            
            logger.info(f"Extracting triples from text ({len(text)} chars) using {self.model}")
            
            # Call Hochschul-LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert knowledge graph extraction system. Extract accurate RDF triples from the provided text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse the response
            extraction_result = self._parse_extraction_response(
                response, text, processing_time
            )
            
            logger.info(f"Triple extraction completed: {extraction_result.triple_count} triples "
                       f"(avg confidence: {extraction_result.average_confidence:.2f})")
            
            return extraction_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Triple extraction failed: {e}", exc_info=True)
            
            return ExtractionResult(
                triples=[],
                source_text=text,
                processing_time_seconds=processing_time,
                model_used=self.model,
                confidence_scores={},
                metadata={"error_type": type(e).__name__},
                success=False,
                error_message=str(e)
            )
    
    def _build_triple_extraction_prompt(
        self,
        text: str,
        domain_context: Optional[str] = None,
        ontology_hints: Optional[List[str]] = None
    ) -> str:
        """Build the prompt for triple extraction"""
        
        prompt_parts = [
            "Extract RDF triples from the following text. Return the results in JSON format.",
            "",
            "Required JSON structure:",
            "```json",
            "{",
            '  "triples": [',
            '    {',
            '      "subject": "entity or concept",',
            '      "predicate": "relationship or property", ',
            '      "object": "related entity, value, or concept",',
            '      "confidence": 0.95,',
            '      "context": "relevant surrounding text"',
            '    }',
            '  ],',
            '  "metadata": {',
            '    "extraction_approach": "description of extraction strategy",',
            '    "domain_detected": "detected domain or topic",',
            '    "quality_indicators": ["list", "of", "quality", "signals"]',
            '  }',
            "}",
            "```",
            "",
            "Guidelines for triple extraction:",
            "- Extract factual relationships, not opinions or speculation",
            "- Use clear, specific predicates (e.g., 'hasAuthor', 'locatedIn', 'publishedOn')",
            "- Ensure subjects and objects are well-defined entities",
            "- Assign confidence scores based on text clarity and evidence",
            "- Include relevant context for each triple",
            "- Focus on the most important and verifiable relationships",
        ]
        
        # Add domain context if provided
        if domain_context:
            prompt_parts.extend([
                "",
                f"Domain context: {domain_context}",
                "Consider domain-specific relationships and terminology."
            ])
        
        # Add ontology hints if provided
        if ontology_hints:
            prompt_parts.extend([
                "",
                "Preferred relationship types:",
                *[f"- {hint}" for hint in ontology_hints],
                "Prioritize these relationship types when applicable."
            ])
        
        prompt_parts.extend([
            "",
            "Text to analyze:",
            "```",
            text,
            "```",
            "",
            "Extract triples now:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_extraction_response(
        self,
        response,
        source_text: str,
        processing_time: float
    ) -> ExtractionResult:
        """Parse the response from Hochschul-LLM"""
        
        try:
            # Get response content
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from Hochschul-LLM")
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    if json_end != -1:
                        data = json.loads(content[json_start:json_end].strip())
                    else:
                        raise
                else:
                    raise
            
            # Extract triples
            triples = []
            triple_data_list = data.get("triples", [])
            
            for triple_data in triple_data_list:
                try:
                    triple = Triple(
                        subject=triple_data["subject"],
                        predicate=triple_data["predicate"],
                        object=triple_data["object"],
                        confidence=triple_data.get("confidence", 0.0),
                        source_chunk=source_text[:200] + "..." if len(source_text) > 200 else source_text,
                        context=triple_data.get("context", ""),
                        metadata=triple_data.get("metadata", {})
                    )
                    triples.append(triple)
                except KeyError as e:
                    logger.warning(f"Skipping malformed triple: missing {e}")
                    continue
            
            # Extract metadata
            metadata = data.get("metadata", {})
            
            # Calculate confidence scores
            confidence_scores = {
                "overall": sum(t.confidence for t in triples) / len(triples) if triples else 0.0,
                "min": min((t.confidence for t in triples), default=0.0),
                "max": max((t.confidence for t in triples), default=0.0)
            }
            
            # Add usage information if available
            if response.usage:
                metadata["token_usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return ExtractionResult(
                triples=triples,
                source_text=source_text,
                processing_time_seconds=processing_time,
                model_used=response.model,
                confidence_scores=confidence_scores,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return ExtractionResult(
                triples=[],
                source_text=source_text,
                processing_time_seconds=processing_time,
                model_used=self.model,
                confidence_scores={},
                metadata={"parse_error": str(e)},
                success=False,
                error_message=f"Response parsing failed: {str(e)}"
            )
    
    async def extract_triples_batch(
        self,
        text_chunks: List[str],
        domain_context: Optional[str] = None,
        ontology_hints: Optional[List[str]] = None
    ) -> List[ExtractionResult]:
        """
        Extract triples from multiple text chunks in batch
        
        Args:
            text_chunks: List of text chunks for processing
            domain_context: Optional domain-specific context
            ontology_hints: Optional ontology relationships to focus on
            
        Returns:
            List of ExtractionResult objects
        """
        logger.info(f"Starting batch triple extraction for {len(text_chunks)} chunks")
        
        # Process in batches to manage API rate limits
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.extract_triples(chunk, domain_context, ontology_hints)
                for chunk in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for chunk, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch extraction failed for chunk: {result}")
                        # Create failed result
                        results.append(ExtractionResult(
                            triples=[],
                            source_text=chunk,
                            processing_time_seconds=0.0,
                            model_used=self.model,
                            confidence_scores={},
                            metadata={"batch_error": str(result)},
                            success=False,
                            error_message=str(result)
                        ))
                    else:
                        results.append(result)
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(text_chunks):
                    await asyncio.sleep(self.config.retry_delay_seconds)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add failed results for the entire batch
                for chunk in batch:
                    results.append(ExtractionResult(
                        triples=[],
                        source_text=chunk,
                        processing_time_seconds=0.0,
                        model_used=self.model,
                        confidence_scores={},
                        metadata={"batch_processing_error": str(e)},
                        success=False,
                        error_message=f"Batch processing failed: {str(e)}"
                    ))
        
        successful_results = sum(1 for r in results if r.success)
        total_triples = sum(r.triple_count for r in results)
        
        logger.info(f"Batch extraction completed: {successful_results}/{len(results)} successful, "
                   f"{total_triples} total triples extracted")
        
        return results
    
    async def validate_triples(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Validate extracted triples for quality and consistency
        
        Args:
            triples: List of triples to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            "total_triples": len(triples),
            "valid_triples": 0,
            "issues": [],
            "quality_score": 0.0,
            "recommendations": []
        }
        
        valid_triples = 0
        
        for i, triple in enumerate(triples):
            issues = []
            
            # Check for empty fields
            if not triple.subject.strip():
                issues.append("Empty subject")
            if not triple.predicate.strip():
                issues.append("Empty predicate")
            if not triple.object.strip():
                issues.append("Empty object")
            
            # Check confidence score
            if triple.confidence < self.config.confidence_threshold:
                issues.append(f"Low confidence: {triple.confidence}")
            
            # Check for generic/vague terms
            generic_terms = ["thing", "entity", "item", "stuff", "something"]
            if any(term in triple.subject.lower() for term in generic_terms):
                issues.append("Generic subject term")
            
            if not issues:
                valid_triples += 1
            else:
                validation_report["issues"].append({
                    "triple_index": i,
                    "triple": triple.to_dict(),
                    "issues": issues
                })
        
        validation_report["valid_triples"] = valid_triples
        validation_report["quality_score"] = valid_triples / len(triples) if triples else 0.0
        
        # Generate recommendations
        if validation_report["quality_score"] < 0.8:
            validation_report["recommendations"].append("Consider adjusting extraction prompt for better quality")
        
        if any("Low confidence" in str(issue) for issue in validation_report["issues"]):
            validation_report["recommendations"].append("Review confidence threshold settings")
        
        return validation_report