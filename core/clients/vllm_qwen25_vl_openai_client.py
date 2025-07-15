"""
OpenAI-compatible vLLM Qwen2.5-VL Client

Uses vLLM's OpenAI-compatible API for Qwen2.5-VL deployment.
This is the recommended approach for Qwen2.5-VL with vLLM.
"""

import base64
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from core.parsers.interfaces.data_models import VisualElementType

logger = logging.getLogger(__name__)


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


class VLLMQwen25VLOpenAIClient:
    """
    OpenAI-compatible vLLM Qwen2.5-VL Client
    
    Uses vLLM server with OpenAI-compatible API for Qwen2.5-VL.
    Requires vLLM > 0.7.2 and a running vLLM server.
    """
    
    MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_retries: int = 3,
        timeout: int = 60,
        auto_start_server: bool = False,
        server_port: int = 8000
    ):
        """
        Initialize the OpenAI-compatible client
        
        Args:
            base_url: vLLM server URL
            api_key: API key (usually "EMPTY" for local deployment)
            max_retries: Number of retries for API calls
            timeout: Request timeout in seconds
            auto_start_server: Whether to automatically start vLLM server
            server_port: Port for vLLM server if auto-starting
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.server_port = server_port
        self.server_process = None
        
        if auto_start_server:
            self._start_vllm_server()
        
        # Wait for server to be ready
        self._wait_for_server()
        
        logger.info(f"Initialized VLLMQwen25VLOpenAIClient with base_url={base_url}")
    
    def _start_vllm_server(self):
        """Start vLLM server process"""
        cmd = [
            "vllm", "serve", self.MODEL_NAME,
            "--port", str(self.server_port),
            "--host", "0.0.0.0",
            "--dtype", "bfloat16",
            "--limit-mm-per-prompt", "image=5,video=5",
            "--max-model-len", "8192",
            "--enable-chunked-prefill"
        ]
        
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info("vLLM server process started")
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            raise
    
    def _wait_for_server(self, max_wait: int = 120):
        """Wait for vLLM server to be ready"""
        start_time = time.time()
        health_url = f"{self.base_url}/health"
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server not ready after {max_wait} seconds")
    
    def _preprocess_image(self, image_data: bytes, max_size: int = 1024) -> str:
        """Convert image bytes to base64 data URL"""
        try:
            # Open and preprocess image
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize if too large
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            output = BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            
            # Convert to base64
            image_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            # Return original as base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
    
    def _build_messages(
        self,
        image_url: str,
        element_type: Optional[VisualElementType] = None,
        analysis_focus: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """Build chat messages for the API request"""
        
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
            {"role": "system", "content": "You are a helpful visual analysis assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
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
        Analyze a visual element using Qwen2.5-VL via OpenAI API
        
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
            # Preprocess image to base64
            image_url = self._preprocess_image(image_data)
            
            # Build messages
            messages = self._build_messages(image_url, element_type, analysis_focus)
            
            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.MODEL_NAME,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            logger.info(f"Sending request to vLLM OpenAI API (focus: {analysis_focus})")
            
            # Make API request with retries
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        break
                    else:
                        logger.warning(f"API request failed (attempt {attempt + 1}): {response.status_code}")
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(f"API request failed: {response.text}")
                        time.sleep(2 ** attempt)
                        
                except requests.exceptions.Timeout:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            
            # Parse response
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Try to parse JSON response
            try:
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    # Try to find JSON in the content
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        json_str = content
                
                data = json.loads(json_str)
                
                description = data.get("description", content)
                confidence = float(data.get("confidence", 0.8))
                ocr_text = data.get("ocr_text")
                extracted_data = data.get("extracted_data")
                element_type_str = data.get("element_type", "unknown_visual")
                
            except (json.JSONDecodeError, ValueError):
                # Fallback to plain text
                description = content
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
                    "model": self.MODEL_NAME,
                    "analysis_focus": analysis_focus,
                    "api_type": "openai_compatible"
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
        """Check if the vLLM service is healthy"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "model": self.MODEL_NAME,
                "api_type": "openai_compatible",
                "base_url": self.base_url,
                "response_code": response.status_code
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.MODEL_NAME,
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup resources and stop server if started"""
        if self.server_process:
            logger.info("Stopping vLLM server")
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
            self.server_process = None