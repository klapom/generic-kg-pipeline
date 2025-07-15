"""
Transformers-based Qwen2.5-VL Client

Uses Hugging Face Transformers directly for Qwen2.5-VL inference.
This is a fallback option when vLLM is not available or has issues.
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

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from core.parsers.interfaces.data_models import VisualElementType, VisualAnalysisResult

logger = logging.getLogger(__name__)


class TransformersQwen25VLClient:
    """
    Transformers-based Qwen2.5-VL Client
    
    Uses Hugging Face Transformers directly for inference.
    Requires transformers >= 4.53.2 and accelerate.
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    def __init__(
        self,
        model_name: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the Transformers client
        
        Args:
            model_name: Model name/path (defaults to Qwen2.5-VL-7B)
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
            
            # Load model with appropriate quantization
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": self.device_map,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _build_messages(
        self,
        image: Image.Image,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """Build chat messages for the model"""
        
        # Build analysis prompt
        prompt_parts = []
        
        if analysis_focus == "comprehensive":
            prompt_parts.extend([
                "Analyze this image and provide a detailed description.",
                "Include:",
                "- What you see in the image",
                "- Any text or numbers visible",
                "- Layout and composition",
                "- Key features and details"
            ])
        elif analysis_focus == "ocr":
            prompt_parts.extend([
                "Extract all text from this image.",
                "Preserve the layout and formatting.",
                "Include all readable text, numbers, and symbols."
            ])
        elif analysis_focus == "chart_data":
            prompt_parts.extend([
                "Analyze this chart or graph.",
                "Extract:",
                "- Chart type",
                "- Data values and labels",
                "- Axes information",
                "- Key trends or insights"
            ])
        else:  # description
            prompt_parts.append("Describe what you see in this image.")
        
        if element_type:
            prompt_parts.append(f"\nHint: This appears to be a {element_type.value}.")
        
        prompt_parts.append("\nProvide your response in JSON format with keys: description, confidence (0-1), element_type, ocr_text (if any), and extracted_data (if applicable).")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "\n".join(prompt_parts)}
                ]
            }
        ]
        
        return messages
    
    def analyze_visual(
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
            # Load image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Build messages
            messages = self._build_messages(image, element_type, analysis_focus)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Extract images and videos from messages
            image_inputs, video_inputs = self._process_vision_info(messages)
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate
            logger.info(f"Generating response (focus: {analysis_focus})...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0
                )
            
            # Decode only the generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse response
            try:
                if "```json" in output_text:
                    json_start = output_text.find("```json") + 7
                    json_end = output_text.find("```", json_start)
                    json_str = output_text[json_start:json_end].strip()
                else:
                    # Try to find JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        json_str = output_text
                
                data = json.loads(json_str)
                
                description = data.get("description", output_text)
                confidence = float(data.get("confidence", 0.8))
                ocr_text = data.get("ocr_text")
                extracted_data = data.get("extracted_data")
                element_type_str = data.get("element_type", "unknown_visual")
                
            except (json.JSONDecodeError, ValueError):
                # Fallback to plain text
                description = output_text
                confidence = 0.7
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
    
    def _process_vision_info(self, messages):
        """Extract images and videos from messages"""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "image":
                        image_inputs.append(item["image"])
                    elif item["type"] == "video":
                        video_inputs.append(item["video"])
        
        return image_inputs, video_inputs
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the model is loaded and ready"""
        try:
            return {
                "status": "healthy" if self.model and self.processor else "unhealthy",
                "model": self.model_name,
                "backend": "transformers",
                "device": str(self.model.device) if self.model else "unknown",
                "dtype": str(self.torch_dtype)
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