"""
Transformers-based Pixtral Client

Uses Hugging Face Transformers for Pixtral-12B inference.
Pixtral is Mistral AI's vision-language model.
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
from transformers import AutoProcessor, LlavaForConditionalGeneration

from core.parsers.interfaces.data_models import VisualElementType, VisualAnalysisResult

logger = logging.getLogger(__name__)


class TransformersPixtralClient:
    """
    Transformers-based Pixtral Client
    
    Uses Hugging Face Transformers for Pixtral-12B inference.
    """
    
    MODEL_NAME = "mistral-community/pixtral-12b"
    
    def __init__(
        self,
        model_name: str = None,
        torch_dtype: torch.dtype = torch.float16,  # Changed from bfloat16 to float16
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the Transformers client
        
        Args:
            model_name: Model name/path (defaults to Pixtral-12B)
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
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
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate quantization
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _build_chat(
        self,
        image: Image.Image,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """Build chat template for Pixtral"""
        
        # Build JSON prompt based on analysis focus
        text_prompt = ""
        
        if analysis_focus == "comprehensive":
            # More specific and detailed prompt for Pixtral
            text_prompt = """You are an expert image analyst. Carefully examine this image and provide a thorough analysis.

IMPORTANT: 
- Be SPECIFIC and DETAILED in your description
- DO NOT give generic responses like "an image of a landscape"
- Identify specific objects, brands, text, colors, layouts, and any notable details
- If it's a car, mention the make, model, color, license plate if visible
- If it's a document page, describe the layout, sections, and key content
- If there's text, transcribe it accurately

Return your analysis as a JSON object with this EXACT structure:
{
  "description": "A comprehensive and specific description of everything you observe in the image, including colors, objects, text, spatial relationships, and any notable details",
  "confidence": 0.95,
  "element_type": "image/chart/diagram/table/screenshot",
  "ocr_text": "any text visible in the image, transcribed exactly as it appears, or null if no text",
  "extracted_data": {"relevant_key": "extracted values, measurements, or structured information"}
}

Example of GOOD description: "A blue BMW 3 Series sedan (G20 model) photographed from the front-left angle on a wet road with mountains in the background. The car has LED headlights, a prominent kidney grille, and license plate 'M-SY 5173'. The image appears to be a promotional photo."

Example of BAD description: "An image of a car on a road."

Now analyze the image and return ONLY the JSON object:"""

        elif analysis_focus == "ocr":
            text_prompt = """Extract all text from this image and return as JSON:
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
        
        # Pixtral chat format - use 'content' for text as per official docs
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": text_prompt},
                    {"type": "image"}  # Image will be passed separately
                ]
            }
        ]
        
        return chat
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from Pixtral response with robust error handling"""
        
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
        Analyze a visual element using Pixtral
        
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
            
            # Build chat
            chat = self._build_chat(image, element_type, analysis_focus)
            
            # Create prompt text first
            prompt = self.processor.apply_chat_template(chat, add_generation_prompt=True)
            logger.debug(f"Generated prompt: {prompt[:300]}...")
            
            # Process text and image together (as per official docs)
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
            
            # Debug input shapes
            if hasattr(inputs, 'pixel_values'):
                logger.debug(f"Pixel values shape: {inputs.pixel_values.shape}, dtype: {inputs.pixel_values.dtype}")
            if hasattr(inputs, 'input_ids'):
                logger.debug(f"Input IDs shape: {inputs.input_ids.shape}")
            
            # Fix dtype mismatch - ensure pixel values match model dtype
            if hasattr(inputs, 'pixel_values'):
                # Check the actual model dtype by looking at the vision tower
                if hasattr(self.model, 'vision_tower'):
                    model_dtype = next(self.model.vision_tower.parameters()).dtype
                else:
                    model_dtype = next(self.model.parameters()).dtype
                    
                logger.debug(f"Model dtype: {model_dtype}")
                logger.debug(f"Pixel values dtype before conversion: {inputs.pixel_values.dtype}")
                
                # Convert pixel values to match model dtype
                inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
                logger.debug(f"Pixel values dtype after conversion: {inputs['pixel_values'].dtype}")
            
            logger.info(f"Generating response (focus: {analysis_focus})...")
            
            # Generate with higher temperature for more detailed responses
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.3,  # Lower temperature to reduce hallucinations
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05  # Slight penalty for repetition
                )
            
            # Decode only the generated tokens
            generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            logger.info(f"Raw output: {output_text}")
            
            # Clean output by removing assistant prefix if present
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