"""Qwen2.5-VL client for visual analysis and description"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel

from core.config import get_config
from plugins.parsers.base_parser import VisualElementType

logger = logging.getLogger(__name__)


class VisualAnalysisConfig(BaseModel):
    """Configuration for visual analysis"""
    model: str = "qwen2.5-vl-72b"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    batch_size: int = 3
    max_image_size: int = 1024  # Max dimension in pixels
    image_quality: int = 85  # JPEG quality for compression


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis from Qwen2.5-VL"""
    description: str
    confidence: float
    extracted_data: Optional[Dict[str, Any]] = None
    element_type_detected: Optional[VisualElementType] = None
    ocr_text: Optional[str] = None
    metadata: Dict[str, Any] = None
    processing_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Qwen25VLClient:
    """
    Client for Qwen2.5-VL visual analysis
    
    Handles image/chart/diagram analysis using Qwen2.5-VL model
    for multi-modal document understanding
    """
    
    def __init__(self, config: Optional[VisualAnalysisConfig] = None):
        """Initialize the Qwen2.5-VL client"""
        self.config = config or VisualAnalysisConfig()
        
        # Get configuration - reuse Hochschul-LLM endpoint for Qwen2.5-VL
        system_config = get_config()
        
        if not system_config.llm.hochschul:
            raise ValueError("Hochschul-LLM configuration not found. Qwen2.5-VL uses the same endpoint.")
        
        # Initialize OpenAI client with Qwen2.5-VL endpoint
        self.client = AsyncOpenAI(
            api_key=system_config.llm.hochschul.api_key,
            base_url=system_config.llm.hochschul.endpoint,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries
        )
        
        self.endpoint = system_config.llm.hochschul.endpoint
        
        logger.info(f"Initialized Qwen2.5-VL client: {self.endpoint}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.close()
    
    def _preprocess_image(self, image_data: bytes) -> bytes:
        """Preprocess image for optimal VLM analysis"""
        try:
            # Open image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = self.config.max_image_size
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save as JPEG with specified quality
            output = BytesIO()
            image.save(output, format='JPEG', quality=self.config.image_quality, optimize=True)
            
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_data  # Return original if preprocessing fails
    
    async def analyze_visual(
        self,
        image_data: bytes,
        document_context: Optional[Dict[str, Any]] = None,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> VisualAnalysisResult:
        """
        Analyze a visual element using Qwen2.5-VL
        
        Args:
            image_data: Raw image bytes
            document_context: Context about the document
            element_type: Hint about the type of visual element
            analysis_focus: Type of analysis (comprehensive, ocr, chart_data, description)
            
        Returns:
            VisualAnalysisResult with analysis details
        """
        start_time = datetime.now()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            image_base64 = base64.b64encode(processed_image).decode('utf-8')
            
            # Build analysis prompt
            prompt = self._build_visual_analysis_prompt(
                document_context, element_type, analysis_focus
            )
            
            logger.info(f"Analyzing visual element with Qwen2.5-VL (focus: {analysis_focus})")
            
            # Call Qwen2.5-VL
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse the response
            analysis_result = self._parse_visual_analysis_response(
                response, processing_time, element_type
            )
            
            logger.info(f"Visual analysis completed: {analysis_result.description[:100]}... "
                       f"(confidence: {analysis_result.confidence:.2f})")
            
            return analysis_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Visual analysis failed: {e}", exc_info=True)
            
            return VisualAnalysisResult(
                description=f"Analysis failed: {str(e)}",
                confidence=0.0,
                processing_time_seconds=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _build_visual_analysis_prompt(
        self,
        document_context: Optional[Dict[str, Any]] = None,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> str:
        """Build the prompt for visual analysis"""
        
        prompt_parts = [
            "Analyze this visual element and provide a detailed description in JSON format.",
            "",
            "Required JSON structure:",
            "```json",
            "{",
            '  "description": "Detailed description of the visual content",',
            '  "element_type": "image|chart|diagram|graph|table|screenshot|drawing|map",',
            '  "confidence": 0.95,',
            '  "ocr_text": "Any readable text in the image",',
            '  "extracted_data": {',
            '    "chart_type": "bar|line|pie|scatter|etc",',
            '    "data_points": ["array", "of", "values"],',
            '    "labels": ["x-axis", "y-axis", "legend"],',
            '    "key_insights": ["insight1", "insight2"]',
            '  },',
            '  "metadata": {',
            '    "colors_used": ["color1", "color2"],',
            '    "layout": "description of layout",',
            '    "quality_assessment": "high|medium|low"',
            '  }',
            "}",
            "```",
            "",
            "Analysis Guidelines:",
        ]
        
        # Add analysis focus specific instructions
        if analysis_focus == "comprehensive":
            prompt_parts.extend([
                "- Provide a complete description of all visual elements",
                "- Extract any text, data, or structured information",
                "- Identify the type and purpose of the visual element",
                "- Note any relationships or patterns in the data"
            ])
        elif analysis_focus == "ocr":
            prompt_parts.extend([
                "- Focus primarily on extracting readable text",
                "- Preserve the layout and structure of text",
                "- Include formatting information if relevant"
            ])
        elif analysis_focus == "chart_data":
            prompt_parts.extend([
                "- Focus on extracting structured data from charts/graphs",
                "- Identify data series, values, and trends",
                "- Extract axis labels, legends, and data points",
                "- Provide insights about the data trends"
            ])
        else:  # description
            prompt_parts.extend([
                "- Provide a clear, detailed description of what is shown",
                "- Focus on the visual content and its meaning",
                "- Explain the context and purpose if apparent"
            ])
        
        # Add element type hint if provided
        if element_type:
            prompt_parts.extend([
                "",
                f"Element type hint: This appears to be a {element_type.value}",
                "Consider this hint but verify based on the actual visual content."
            ])
        
        # Add document context if provided
        if document_context:
            prompt_parts.extend([
                "",
                "Document Context:"
            ])
            
            if "document_title" in document_context:
                prompt_parts.append(f"- Document: {document_context['document_title']}")
            if "section" in document_context:
                prompt_parts.append(f"- Section: {document_context['section']}")
            if "surrounding_text" in document_context:
                prompt_parts.append(f"- Context: {document_context['surrounding_text']}")
            if "document_type" in document_context:
                prompt_parts.append(f"- Type: {document_context['document_type']}")
            
            prompt_parts.append("Use this context to provide more relevant and accurate analysis.")
        
        prompt_parts.extend([
            "",
            "Provide accurate, detailed analysis in the JSON format above:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_visual_analysis_response(
        self,
        response,
        processing_time: float,
        suggested_type: Optional[VisualElementType] = None
    ) -> VisualAnalysisResult:
        """Parse the response from Qwen2.5-VL"""
        
        try:
            # Get response content
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from Qwen2.5-VL")
            
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
            
            # Extract analysis data
            description = data.get("description", "No description provided")
            confidence = float(data.get("confidence", 0.0))
            ocr_text = data.get("ocr_text")
            extracted_data = data.get("extracted_data")
            metadata = data.get("metadata", {})
            
            # Determine element type
            element_type_str = data.get("element_type", "unknown_visual")
            element_type_detected = None
            
            try:
                # Try to map string to VisualElementType
                for vtype in VisualElementType:
                    if vtype.value == element_type_str:
                        element_type_detected = vtype
                        break
                
                if not element_type_detected:
                    element_type_detected = VisualElementType.UNKNOWN_VISUAL
                    
            except Exception:
                element_type_detected = suggested_type or VisualElementType.UNKNOWN_VISUAL
            
            # Add usage information if available
            if response.usage:
                metadata["token_usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return VisualAnalysisResult(
                description=description,
                confidence=confidence,
                extracted_data=extracted_data,
                element_type_detected=element_type_detected,
                ocr_text=ocr_text,
                metadata=metadata,
                processing_time_seconds=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to parse visual analysis response: {e}")
            return VisualAnalysisResult(
                description=f"Response parsing failed: {str(e)}",
                confidence=0.0,
                processing_time_seconds=processing_time,
                success=False,
                error_message=f"Response parsing failed: {str(e)}"
            )
    
    async def analyze_visuals_batch(
        self,
        visual_data_list: List[bytes],
        document_context: Optional[Dict[str, Any]] = None,
        element_types: Optional[List[VisualElementType]] = None,
        analysis_focus: str = "comprehensive"
    ) -> List[VisualAnalysisResult]:
        """
        Analyze multiple visual elements in batch
        
        Args:
            visual_data_list: List of image bytes to analyze
            document_context: Context about the document
            element_types: Optional hints about element types
            analysis_focus: Type of analysis to perform
            
        Returns:
            List of VisualAnalysisResult objects
        """
        logger.info(f"Starting batch visual analysis for {len(visual_data_list)} elements")
        
        # Process in batches to manage API rate limits
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(visual_data_list), batch_size):
            batch_data = visual_data_list[i:i + batch_size]
            batch_types = element_types[i:i + batch_size] if element_types else [None] * len(batch_data)
            
            # Process batch concurrently
            batch_tasks = [
                self.analyze_visual(
                    data, 
                    document_context, 
                    element_type, 
                    analysis_focus
                )
                for data, element_type in zip(batch_data, batch_types)
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Batch visual analysis failed: {result}")
                        # Create failed result
                        results.append(VisualAnalysisResult(
                            description=f"Analysis failed: {str(result)}",
                            confidence=0.0,
                            success=False,
                            error_message=str(result)
                        ))
                    else:
                        results.append(result)
                
                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(visual_data_list):
                    await asyncio.sleep(self.config.retry_delay_seconds)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add failed results for the entire batch
                for _ in batch_data:
                    results.append(VisualAnalysisResult(
                        description=f"Batch processing failed: {str(e)}",
                        confidence=0.0,
                        success=False,
                        error_message=f"Batch processing failed: {str(e)}"
                    ))
        
        successful_results = sum(1 for r in results if r.success)
        
        logger.info(f"Batch visual analysis completed: {successful_results}/{len(results)} successful")
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the Qwen2.5-VL service is healthy
        
        Returns:
            Health status information
        """
        try:
            start_time = datetime.now()
            
            # Create a simple test image (1x1 pixel)
            test_image = Image.new('RGB', (1, 1), color='white')
            test_image_bytes = BytesIO()
            test_image.save(test_image_bytes, format='JPEG')
            test_image_data = test_image_bytes.getvalue()
            
            # Test with a simple visual analysis request
            test_base64 = base64.b64encode(test_image_data).decode('utf-8')
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this test image briefly. Respond with: {'description': 'test response', 'confidence': 1.0}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{test_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check if we got a valid response
            if response.choices and response.choices[0].message.content:
                return {
                    "status": "healthy",
                    "endpoint": self.endpoint,
                    "model": self.config.model,
                    "response_time_ms": response_time,
                    "test_response": response.choices[0].message.content.strip(),
                    "capabilities": ["image_analysis", "chart_analysis", "ocr", "visual_description"],
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
            logger.error(f"Qwen2.5-VL health check failed: {e}")
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }