"""
Transformers-based LLaVA Client

Uses Hugging Face Transformers for LLaVA-1.6 inference.
LLaVA (Large Language and Vision Assistant) is a powerful multimodal model.
"""

import base64
import gc
import json
import logging
import torch
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional
import re

from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from core.parsers.interfaces.data_models import VisualElementType, VisualAnalysisResult

logger = logging.getLogger(__name__)


class TransformersLLaVAClient:
    """
    Transformers-based LLaVA Client
    
    Uses Hugging Face Transformers for LLaVA-1.6 inference.
    """
    
    MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"  # Can also use "llava-hf/llava-v1.6-mistral-7b-hf"
    
    def __init__(
        self,
        model_name: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True  # Default to 4-bit for 34B model
    ):
        """
        Initialize the Transformers client
        
        Args:
            model_name: Model name/path (defaults to LLaVA-1.6-34B)
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization (recommended for 34B)
        """
        self.model_name = model_name or self.MODEL_NAME
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and processor"""
        try:
            logger.info(f"Loading {self.model_name} with Transformers...")
            
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate quantization
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _build_conversation(
        self,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """Build conversation for LLaVA-1.6 using chat template"""
        
        # Build the text prompt based on analysis focus
        text_prompt = ""
        
        if analysis_focus == "comprehensive":
            text_prompt = """Analyze this image and return ONLY a valid JSON object with this structure:
{
  "description": "detailed description of what you see",
  "confidence": 0.95,
  "element_type": "image/chart/diagram/table/screenshot",
  "ocr_text": "any text visible in the image or null",
  "extracted_data": {"key": "value for any structured data or null"}
}

Important: Return ONLY the JSON object, no explanations before or after."""

        elif analysis_focus == "ocr":
            text_prompt = """Extract all text from this image and return it as a JSON object:
{
  "description": "Image contains text content",
  "confidence": 0.90,
  "element_type": "text",
  "ocr_text": "all extracted text exactly as it appears",
  "extracted_data": null
}

Return ONLY the JSON object."""

        elif analysis_focus == "chart_data":
            text_prompt = """Analyze this chart/graph and return ONLY a JSON object:
{
  "description": "description of the chart type and what it shows",
  "confidence": 0.88,
  "element_type": "chart",
  "ocr_text": "any text labels or values visible",
  "extracted_data": {"chart_type": "bar/line/pie/etc", "data_points": [], "insights": ""}
}

Return ONLY the JSON object."""

        else:  # description
            text_prompt = """Describe what you see in this image and return as JSON:
{
  "description": "detailed description",
  "confidence": 0.85,
  "element_type": "image",
  "ocr_text": null,
  "extracted_data": null
}

Return ONLY the JSON object."""
        
        if element_type:
            text_prompt += f"\n\nHint: This appears to be a {element_type.value}."
        
        # LLaVA-1.6 conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ],
            },
        ]
        
        return conversation
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLaVA response with robust error handling"""
        
        # Clean the response
        response = response.strip()
        
        # Method 1: Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code blocks
        if "```json" in response:
            try:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    json_str = response[json_start:json_end].strip()
                    return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Method 3: Find JSON-like patterns
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object (no nested objects)
            r'\{.*?\}',     # Non-greedy object match
            r'\{.*\}',      # Greedy object match (last resort)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Try the largest match first (most complete)
                for match in sorted(matches, key=len, reverse=True):
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue
        
        # Method 4: Try to clean common issues and parse
        cleaned_response = response
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "Here's the JSON:",
            "The JSON output is:",
            "JSON:",
            "Response:",
            "Answer:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()
        
        # Remove markdown code blocks
        cleaned_response = re.sub(r'```(?:json)?\s*', '', cleaned_response)
        
        # Try parsing the cleaned response
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            pass
        
        # Method 5: If all else fails, construct a basic JSON from the text
        logger.warning(f"Could not extract JSON, attempting to construct from text: {response[:100]}...")
        
        # Try to extract basic information even if not in JSON format
        try:
            return {
                "description": response,
                "confidence": 0.7,
                "element_type": "unknown_visual",
                "ocr_text": None,
                "extracted_data": None
            }
        except Exception:
            return None
    
    def analyze_visual(
        self,
        image_data: bytes,
        document_context: Optional[Dict[str, Any]] = None,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> VisualAnalysisResult:
        """
        Analyze a visual element using LLaVA
        
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
            # Load image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Build conversation
            conversation = self._build_conversation(element_type, analysis_focus)
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True
            )
            
            logger.info(f"Generated prompt preview: {prompt[:200]}...")
            
            # Process inputs
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.model.device)
            
            # Generate
            logger.info(f"Generating response (focus: {analysis_focus})...")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.2,  # Lower temperature for JSON consistency
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens (exclude input)
            generated_ids = output[0][inputs.input_ids.shape[1]:]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"Raw output: {output_text}")
            
            # Extract assistant response
            if "ASSISTANT:" in output_text:
                output_text = output_text.split("ASSISTANT:")[-1].strip()
            
            logger.info(f"Cleaned output: {output_text}")
            
            # Parse response with improved JSON extraction
            parsed_data = self._extract_json_from_response(output_text)
            
            if parsed_data:
                logger.info(f"Successfully parsed JSON: {parsed_data}")
                description = parsed_data.get("description", output_text)
                confidence = float(parsed_data.get("confidence", 0.85))
                ocr_text = parsed_data.get("ocr_text")
                extracted_data = parsed_data.get("extracted_data")
                element_type_str = parsed_data.get("element_type", "unknown_visual")
            else:
                logger.warning(f"Failed to parse JSON, using fallback. Raw output: {output_text}")
                # Fallback to plain text
                description = output_text
                confidence = 0.75
                ocr_text = None
                extracted_data = None
                element_type_str = "unknown_visual"
            
            # Map element type
            element_type_detected = None
            try:
                for vtype in VisualElementType:
                    if vtype.value == element_type_str:
                        element_type_detected = vtype
                        break
                if not element_type_detected:
                    element_type_detected = VisualElementType.UNKNOWN_VISUAL
            except Exception:
                element_type_detected = element_type or VisualElementType.UNKNOWN_VISUAL
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VisualAnalysisResult(
                description=description,
                confidence=confidence,
                extracted_data=extracted_data,
                element_type_detected=element_type_detected,
                ocr_text=ocr_text,
                metadata={
                    "processing_time": processing_time,
                    "model": self.model_name,
                    "analysis_focus": analysis_focus,
                    "backend": "transformers"
                },
                processing_time_seconds=processing_time,
                success=True
            )
            
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
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the model is loaded and ready"""
        try:
            return {
                "status": "healthy" if self.model and self.processor else "unhealthy",
                "model": self.model_name,
                "backend": "transformers",
                "device": str(self.model.device) if self.model else "unknown",
                "dtype": str(self.torch_dtype),
                "quantization": "8bit" if self.load_in_8bit else ("4bit" if self.load_in_4bit else "none")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup resources and free GPU memory"""
        if self.model:
            logger.info("Cleaning up model resources")
            del self.model
            self.model = None
        
        if self.processor:
            del self.processor
            self.processor = None
        
        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()