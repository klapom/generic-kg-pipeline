"""
Qwen2.5-VL client for visual analysis and description
Modernized with standardized BaseModelClient architecture
"""

import base64
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel

from core.clients.base import BaseModelClient, BatchProcessingMixin
from core.config_new.unified_manager import get_config
from core.parsers import VisualElementType

logger = logging.getLogger(__name__)


# Data Models
class VisualAnalysisConfig(BaseModel):
    """Configuration for visual analysis"""
    model: str = "qwen2.5-vl-72b"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    batch_size: int = 3
    max_image_size: int = 1024
    image_quality: int = 85


class VisualAnalysisResult(BaseModel):
    """Result of visual analysis from Qwen2.5-VL"""
    description: str
    confidence: float = 0.0
    extracted_data: Optional[Dict[str, Any]] = None
    element_type_detected: Optional[VisualElementType] = None
    ocr_text: Optional[str] = None
    metadata: Dict[str, Any] = {}
    processing_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class VisualAnalysisRequest(BaseModel):
    """Request for visual analysis"""
    image_data: bytes
    document_context: Optional[Dict[str, Any]] = None
    element_type: Optional[VisualElementType] = None
    analysis_focus: str = "comprehensive"  # comprehensive, ocr, chart_data, description
    
    class Config:
        # Allow bytes type
        arbitrary_types_allowed = True


class Qwen25VLClient(BaseModelClient[VisualAnalysisRequest, VisualAnalysisResult, VisualAnalysisConfig],
                    BatchProcessingMixin):
    """
    Modernized client for Qwen2.5-VL visual analysis
    
    Handles image/chart/diagram analysis using Qwen2.5-VL model
    for multi-modal document understanding
    
    Benefits over original:
    - Automatic retry logic for API failures
    - Standardized health checks
    - Built-in metrics collection
    - Batch image processing support
    - Unified error handling
    """
    
    def __init__(self, config: Optional[VisualAnalysisConfig] = None):
        """Initialize with special handling for OpenAI client"""
        self._openai_client = None
        # Use hochschul_llm service config (they share the same endpoint)
        super().__init__("hochschul_llm", config=config)
        
    def _get_default_config(self) -> VisualAnalysisConfig:
        """Default configuration for visual analysis"""
        return VisualAnalysisConfig()
    
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
    
    async def close(self):
        """Close connections"""
        if self._openai_client:
            await self._openai_client.close()
        await super().close()
    
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
    
    async def _process_internal(self, request: VisualAnalysisRequest) -> VisualAnalysisResult:
        """
        Internal visual analysis processing
        
        Args:
            request: Image data and analysis parameters
            
        Returns:
            Analysis result
        """
        self._initialize_openai_client()
        start_time = datetime.now()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(request.image_data)
            image_base64 = base64.b64encode(processed_image).decode('utf-8')
            
            # Build analysis prompt
            prompt = self._build_visual_analysis_prompt(
                request.document_context,
                request.element_type,
                request.analysis_focus
            )
            
            logger.info(f"Analyzing visual element with Qwen2.5-VL (focus: {request.analysis_focus})")
            
            # Call Qwen2.5-VL
            response = await self._openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_json = json.loads(response.choices[0].message.content)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VisualAnalysisResult(
                description=result_json.get("description", ""),
                confidence=result_json.get("confidence", 0.8),
                extracted_data=result_json.get("extracted_data"),
                element_type_detected=self._detect_element_type(result_json),
                ocr_text=result_json.get("ocr_text"),
                metadata={
                    "model": self.config.model,
                    "focus": request.analysis_focus,
                    "image_size": len(processed_image),
                    "tokens_used": response.usage.total_tokens
                },
                processing_time_seconds=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return VisualAnalysisResult(
                description="",
                confidence=0.0,
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=str(e)
            )
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Qwen2.5-VL specific health check"""
        self._initialize_openai_client()
        
        # Create a simple test image (1x1 white pixel)
        test_image = Image.new('RGB', (1, 1), color='white')
        img_buffer = BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Test with simple visual query
        response = await self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }],
            max_tokens=10,
            temperature=0
        )
        
        return {
            "model_available": True,
            "model_name": self.config.model,
            "supports_vision": True,
            "test_response": response.choices[0].message.content
        }
    
    def _build_visual_analysis_prompt(self,
                                    document_context: Optional[Dict[str, Any]],
                                    element_type: Optional[VisualElementType],
                                    analysis_focus: str) -> str:
        """Build the analysis prompt based on focus"""
        prompts = {
            "comprehensive": """Analyze this image comprehensively. Return a JSON object with:
{
    "description": "Detailed description of what you see",
    "confidence": 0.0-1.0,
    "extracted_data": {
        "type": "chart/diagram/photo/table/other",
        "key_elements": ["list of key elements"],
        "relationships": ["list of relationships if any"]
    },
    "ocr_text": "Any text found in the image"
}""",
            
            "ocr": """Extract all text from this image. Return a JSON object with:
{
    "description": "Brief description",
    "confidence": 0.0-1.0,
    "ocr_text": "All extracted text"
}""",
            
            "chart_data": """Extract data from this chart/graph. Return a JSON object with:
{
    "description": "Chart type and what it shows",
    "confidence": 0.0-1.0,
    "extracted_data": {
        "chart_type": "bar/line/pie/etc",
        "title": "chart title",
        "axes": {"x": "label", "y": "label"},
        "data_points": [{"label": "...", "value": ...}]
    }
}""",
            
            "description": """Describe this image for document understanding. Return a JSON object with:
{
    "description": "Clear, informative description",
    "confidence": 0.0-1.0,
    "extracted_data": {
        "main_subject": "what the image primarily shows",
        "context": "how it relates to the document"
    }
}"""
        }
        
        base_prompt = prompts.get(analysis_focus, prompts["comprehensive"])
        
        if document_context:
            base_prompt = f"Document context: {json.dumps(document_context)}\n\n{base_prompt}"
        
        if element_type:
            base_prompt = f"Expected element type: {element_type}\n\n{base_prompt}"
        
        return base_prompt
    
    def _detect_element_type(self, result_json: Dict[str, Any]) -> Optional[VisualElementType]:
        """Detect the visual element type from the result"""
        extracted_data = result_json.get("extracted_data", {})
        detected_type = extracted_data.get("type", "").lower()
        
        type_mapping = {
            "chart": VisualElementType.CHART,
            "graph": VisualElementType.CHART,
            "diagram": VisualElementType.DIAGRAM,
            "table": VisualElementType.TABLE,
            "photo": VisualElementType.IMAGE,
            "image": VisualElementType.IMAGE,
            "figure": VisualElementType.FIGURE,
            "formula": VisualElementType.FORMULA,
            "equation": VisualElementType.FORMULA
        }
        
        return type_mapping.get(detected_type)
    
    async def analyze_visual(self,
                           image_data: bytes,
                           document_context: Optional[Dict[str, Any]] = None,
                           element_type: Optional[VisualElementType] = None,
                           analysis_focus: str = "comprehensive") -> VisualAnalysisResult:
        """
        Analyze a visual element
        
        Args:
            image_data: Raw image bytes
            document_context: Context about the document
            element_type: Hint about the type of visual element
            analysis_focus: Type of analysis
            
        Returns:
            VisualAnalysisResult
        """
        request = VisualAnalysisRequest(
            image_data=image_data,
            document_context=document_context,
            element_type=element_type,
            analysis_focus=analysis_focus
        )
        return await self.process(request)
    
    async def batch_analyze(self,
                          images: List[bytes],
                          analysis_focus: str = "comprehensive") -> List[VisualAnalysisResult]:
        """
        Analyze multiple images in batch
        
        Args:
            images: List of image data
            analysis_focus: Type of analysis for all images
            
        Returns:
            List of analysis results
        """
        requests = [
            VisualAnalysisRequest(image_data=img, analysis_focus=analysis_focus)
            for img in images
        ]
        
        return await self.process_batch(
            requests,
            batch_size=self.config.batch_size,
            concurrent_batches=2
        )


# Example usage
async def example_usage():
    """Show benefits of new architecture"""
    
    async with Qwen25VLClient() as client:
        # 1. Health check
        health = await client.health_check()
        print(f"Service Status: {health.status}")
        
        # 2. Analyze single image with auto-retry
        with open("chart.png", "rb") as f:
            image_data = f.read()
        
        result = await client.analyze_visual(
            image_data,
            analysis_focus="chart_data"
        )
        print(f"Analysis confidence: {result.confidence}")
        
        # 3. Batch process multiple images
        image_files = ["img1.png", "img2.png", "img3.png"]
        images = []
        for img_file in image_files:
            with open(img_file, "rb") as f:
                images.append(f.read())
        
        results = await client.batch_analyze(images)
        print(f"Analyzed {len(results)} images")
        
        # 4. Get metrics
        metrics = client.get_metrics()
        print(f"Average response time: {metrics.average_response_time_ms}ms")