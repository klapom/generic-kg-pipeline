"""
Local vLLM Llama-3.2-11B-Vision Client

High-performance visual analysis using local vLLM Llama-3.2-11B-Vision model.
Supports advanced multimodal understanding and reasoning.
"""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from core.vllm.base_client import BaseVLLMClient, InferenceRequest, InferenceResult
from core.vllm.model_manager import ModelConfig, SamplingConfig
from core.parsers.interfaces.data_models import VisualElementType

logger = logging.getLogger(__name__)


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis from Llama-3.2-11B-Vision"""
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


class VLLMLlama32VisionClient(BaseVLLMClient):
    """
    Local vLLM Llama-3.2-11B-Vision Client
    
    Provides high-performance visual analysis using Meta's Llama-3.2-11B-Vision:
    - Advanced multimodal reasoning
    - Enhanced visual understanding
    - Detailed image descriptions
    - Chart and diagram analysis
    """
    
    MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.8,
        max_image_size: int = 1024,
        image_quality: int = 85,
        batch_size: int = 2,
        auto_load: bool = False
    ):
        # Create model configuration optimized for Llama-3.2-11B-Vision
        model_config = ModelConfig(
            model_name=self.MODEL_NAME,
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=False,
            limit_mm_per_prompt={"image": 4},  # Support multiple images
            dtype="auto"
        )
        
        # Sampling configuration optimized for visual analysis
        sampling_config = SamplingConfig(
            temperature=0.1,
            max_tokens=2000,
            top_p=1.0,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        
        # Llama-3.2 Vision specific configuration
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        self.batch_size = batch_size
        
        super().__init__(
            model_id="llama32_vision",
            model_config=model_config,
            sampling_config=sampling_config,
            auto_load=auto_load
        )
        
        logger.info(f"Initialized VLLMLlama32VisionClient with batch_size={batch_size}")
    
    def get_warmup_prompts(self) -> List[str]:
        """Llama-3.2-Vision specific warmup prompts"""
        return [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nDescribe this test image briefly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnalyze the visual elements in this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
    
    def _get_input_size(self, args, kwargs) -> int:
        """Extract input size for performance tracking"""
        if args and len(args) > 0:
            if isinstance(args[0], list):
                return len(args[0])
        return 1
    
    def _get_output_size(self, result) -> int:
        """Extract output size for performance tracking"""
        if isinstance(result, list):
            return len(result)
        elif isinstance(result, VisualAnalysisResult):
            return 1
        return 0
    
    def _preprocess_image(self, image_data: bytes) -> str:
        """Preprocess image for optimal VLM analysis"""
        try:
            # Open image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize if too large
            if max(image.size) > self.max_image_size:
                image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            
            # Save as JPEG with specified quality
            output = BytesIO()
            image.save(output, format='JPEG', quality=self.image_quality, optimize=True)
            
            # Convert to base64
            image_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            # Return original as base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
    
    def _build_visual_analysis_prompt(
        self,
        document_context: Optional[Dict[str, Any]] = None,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> str:
        """Build the prompt for visual analysis using Llama format"""
        
        # Llama-3.2 uses a specific prompt format with headers
        prompt_parts = [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "You are an expert visual analysis AI. Analyze images and provide detailed, structured responses in JSON format.",
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
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
            "Provide accurate, detailed analysis in the JSON format above:",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ])
        
        return "\n".join(prompt_parts)
    
    def parse_model_output(self, output: Any) -> Dict[str, Any]:
        """Parse Llama-3.2-Vision model output"""
        try:
            # Extract text from vLLM output
            if hasattr(output, 'outputs') and output.outputs:
                content = output.outputs[0].text
            elif hasattr(output, 'text'):
                content = output.text
            else:
                content = str(output)
            
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
                        data = json.loads(content[json_start:].strip())
                else:
                    raise
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from model output: {e}")
            logger.debug(f"Raw output: {content[:500]}...")
            
            # Return fallback structure
            return {
                "description": content[:500] if content else "Analysis failed",
                "element_type": "unknown_visual",
                "confidence": 0.0,
                "ocr_text": None,
                "extracted_data": None,
                "metadata": {"parsing_error": str(e)}
            }
        except Exception as e:
            logger.error(f"Failed to parse model output: {e}")
            return {
                "description": f"Parsing failed: {str(e)}",
                "element_type": "unknown_visual", 
                "confidence": 0.0,
                "ocr_text": None,
                "extracted_data": None,
                "metadata": {"parsing_error": str(e)}
            }
    
    def analyze_visual(
        self,
        image_data: bytes,
        document_context: Optional[Dict[str, Any]] = None,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> VisualAnalysisResult:
        """
        Analyze a visual element using Llama-3.2-11B-Vision
        
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
            # Ensure model is loaded
            logger.info("🦙 Ensuring Llama-3.2-11B-Vision model is loaded in LOCAL vLLM...")
            if not self.ensure_model_loaded():
                raise RuntimeError("Failed to load Llama-3.2-11B-Vision model")
            logger.info("✅ Llama-3.2-11B-Vision model ready in LOCAL vLLM")
            
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            
            # Build analysis prompt
            prompt = self._build_visual_analysis_prompt(
                document_context, element_type, analysis_focus
            )
            
            logger.info(f"🦙 LOCAL vLLM Llama-3.2-Vision: Analyzing visual element (focus: {analysis_focus})")
            
            # Create multimodal request
            multimodal_prompt = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [processed_image]
                }
            }
            
            request = InferenceRequest(
                prompts=[multimodal_prompt],
                sampling_config=self.sampling_config
            )
            
            # Call Llama-3.2-Vision
            result = self.generate(request)
            
            if not result.success:
                raise RuntimeError(f"Llama-3.2-Vision inference failed: {result.error_message}")
            
            if not result.outputs:
                raise RuntimeError("No output from Llama-3.2-Vision model")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse the response
            data = self.parse_model_output(result.outputs[0])
            
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
                element_type_detected = element_type or VisualElementType.UNKNOWN_VISUAL
            
            # Add processing metadata
            metadata.update({
                "processing_time": processing_time,
                "model": self.MODEL_NAME,
                "analysis_focus": analysis_focus
            })
            
            analysis_result = VisualAnalysisResult(
                description=description,
                confidence=confidence,
                extracted_data=extracted_data,
                element_type_detected=element_type_detected,
                ocr_text=ocr_text,
                metadata=metadata,
                processing_time_seconds=processing_time,
                success=True
            )
            
            logger.debug(f"Visual analysis completed: {description[:100]}... "
                        f"(confidence: {confidence:.2f})")
            
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
    
    def analyze_visuals_batch(
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
        if not visual_data_list:
            return []
        
        logger.info(f"Starting batch visual analysis for {len(visual_data_list)} elements")
        
        # Ensure model is loaded once for the entire batch
        if not self.ensure_model_loaded():
            logger.error("Failed to load Llama-3.2-Vision model for batch processing")
            return [VisualAnalysisResult(
                description="Model loading failed",
                confidence=0.0,
                success=False,
                error_message="Model loading failed"
            ) for _ in visual_data_list]
        
        # Process in batches to manage memory
        results = []
        batch_size = self.batch_size
        
        for i in range(0, len(visual_data_list), batch_size):
            batch_data = visual_data_list[i:i + batch_size]
            batch_types = element_types[i:i + batch_size] if element_types else [None] * len(batch_data)
            
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch_data)} images")
            
            # Process batch sequentially (vLLM handles internal optimization)
            batch_results = []
            for data, element_type in zip(batch_data, batch_types):
                try:
                    result = self.analyze_visual(
                        data,
                        document_context,
                        element_type,
                        analysis_focus
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Batch visual analysis failed: {e}")
                    batch_results.append(VisualAnalysisResult(
                        description=f"Analysis failed: {str(e)}",
                        confidence=0.0,
                        success=False,
                        error_message=str(e)
                    ))
            
            results.extend(batch_results)
        
        successful_results = sum(1 for r in results if r.success)
        
        logger.info(f"Batch visual analysis completed: {successful_results}/{len(results)} successful")
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the Llama-3.2-Vision service is healthy
        
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
            result = self.analyze_visual(
                test_image_data,
                analysis_focus="description"
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.success:
                return {
                    "status": "healthy",
                    "model_id": self.model_id,
                    "model_name": self.MODEL_NAME,
                    "response_time_ms": response_time,
                    "test_result": result.description[:100],
                    "capabilities": ["advanced_reasoning", "visual_understanding", "chart_analysis", "ocr"],
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "model_id": self.model_id,
                    "error": result.error_message,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Llama-3.2-Vision health check failed: {e}")
            return {
                "status": "unhealthy",
                "model_id": self.model_id,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

