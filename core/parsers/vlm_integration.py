"""
VLM Integration Module

Handles Visual Language Model integration for analyzing visual elements
and updating visual segments with descriptions. Supports multiple VLM models
for comparative analysis.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from core.parsers.interfaces.data_models import (
    VisualElement, VisualElementType, Segment, 
    SegmentType, VisualSubtype
)
from core.clients.vllm_qwen25_vl_local import VLLMQwen25VLClient, VisualAnalysisResult
from core.clients.vllm_llama32_vision_local import VLLMLlama32VisionClient


class VLMModelType(str, Enum):
    """Supported VLM model types for local deployment"""
    QWEN25_VL_7B = "qwen2.5-vl-7b"
    LLAMA32_VISION_11B = "llama-3.2-11b-vision"
    LLAVA_16_34B = "llava-1.6-34b"
    LLAVA_16_7B = "llava-1.6-7b"  # Smaller variant
    PIXTRAL_12B = "pixtral-12b"


class VLMComparison:
    """Results from comparing multiple VLM models"""
    
    def __init__(self):
        self.model_results: Dict[str, Dict[str, Any]] = {}
        self.consensus_score: float = 0.0
        self.best_model: Optional[str] = None
        self.timestamp = datetime.now().isoformat()
    
    def add_model_result(self, model_name: str, result: VisualAnalysisResult, processing_time: float):
        """Add result from a specific VLM model"""
        self.model_results[model_name] = {
            "description": result.description,
            "confidence": result.confidence,
            "visual_subtype": result.visual_subtype,
            "processing_time_seconds": processing_time,
            "detected_elements": result.detected_elements,
            "extracted_text": result.extracted_text
        }
    
    def calculate_consensus(self) -> Dict[str, Any]:
        """Calculate consensus across all models"""
        if not self.model_results:
            return {}
        
        # Simple consensus: average confidence and most common subtype
        total_confidence = sum(r["confidence"] for r in self.model_results.values())
        avg_confidence = total_confidence / len(self.model_results)
        
        # Find most common visual subtype
        subtypes = [r["visual_subtype"] for r in self.model_results.values()]
        most_common_subtype = max(set(subtypes), key=subtypes.count) if subtypes else None
        
        # Find best model (highest confidence)
        best_model = max(self.model_results.keys(), 
                        key=lambda k: self.model_results[k]["confidence"])
        
        self.consensus_score = avg_confidence
        self.best_model = best_model
        
        return {
            "consensus_confidence": avg_confidence,
            "most_common_subtype": most_common_subtype,
            "best_model": best_model,
            "model_count": len(self.model_results)
        }

logger = logging.getLogger(__name__)


class AdaptivePromptGenerator:
    """
    Generates adaptive prompts based on visual element types
    """
    
    def generate_prompt(
        self, 
        element_type: VisualElementType,
        document_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate element-specific prompts for better VLM analysis
        
        Args:
            element_type: Type of visual element
            document_context: Additional context about the document
            
        Returns:
            Dict with analysis_focus and additional context
        """
        # Base context
        context = document_context.copy() if document_context else {}
        
        # Element-specific prompts
        if element_type == VisualElementType.CHART:
            return {
                "analysis_focus": "chart_data",
                "context": {
                    **context,
                    "instructions": [
                        "Extract all data points and values",
                        "Identify axis labels and units",
                        "Describe the chart type (bar, line, pie, etc.)",
                        "Summarize key trends or insights",
                        "Extract legend information if present"
                    ]
                }
            }
        
        elif element_type == VisualElementType.DIAGRAM:
            return {
                "analysis_focus": "comprehensive",
                "context": {
                    **context,
                    "instructions": [
                        "Describe the diagram structure and components",
                        "Identify relationships between elements",
                        "Extract any text labels or annotations",
                        "Explain the diagram's purpose or meaning",
                        "Note any flow directions or hierarchies"
                    ]
                }
            }
        
        elif element_type == VisualElementType.FORMULA:
            return {
                "analysis_focus": "ocr",
                "context": {
                    **context,
                    "instructions": [
                        "Extract the mathematical formula in LaTeX format",
                        "Identify all variables and symbols",
                        "Preserve subscripts and superscripts",
                        "Note any special mathematical notation",
                        "Provide the formula meaning if apparent"
                    ]
                }
            }
        
        elif element_type == VisualElementType.TABLE or element_type == VisualElementType.TABLE_IMAGE:
            return {
                "analysis_focus": "ocr",
                "context": {
                    **context,
                    "instructions": [
                        "Extract table structure with headers",
                        "Preserve row and column relationships",
                        "Extract all cell values accurately",
                        "Note any merged cells or special formatting",
                        "Identify table caption if present"
                    ]
                }
            }
        
        elif element_type == VisualElementType.SCREENSHOT:
            return {
                "analysis_focus": "comprehensive",
                "context": {
                    **context,
                    "instructions": [
                        "Describe the UI elements visible",
                        "Extract any text content",
                        "Identify the application or system shown",
                        "Note any important data or status indicators",
                        "Describe user interactions if apparent"
                    ]
                }
            }
        
        else:  # Default for IMAGE, DRAWING, MAP, FIGURE, UNKNOWN_VISUAL
            return {
                "analysis_focus": "description",
                "context": {
                    **context,
                    "instructions": [
                        "Provide detailed description of the visual content",
                        "Identify main subjects or objects",
                        "Note colors, composition, and style",
                        "Extract any visible text",
                        "Explain the image's purpose in the document context"
                    ]
                }
            }


class MultiVLMIntegration:
    """
    Multi-VLM comparison integration for comprehensive visual analysis
    """
    
    def __init__(self, enabled_models: List[VLMModelType] = None):
        """
        Initialize multi-VLM integration
        
        Args:
            enabled_models: List of VLM models to use for comparison
        """
        self.enabled_models = enabled_models or [VLMModelType.QWEN25_VL_7B]
        self.vlm_clients: Dict[str, Any] = {}
        self.prompt_generator = AdaptivePromptGenerator()
        self._initialized = False
        
    async def _initialize_clients(self):
        """Initialize all VLM clients"""
        if self._initialized:
            return
            
        logger.info(f"Initializing {len(self.enabled_models)} VLM clients...")
        
        # Note: For multi-model comparison, we'll initialize models one at a time
        # to avoid GPU memory issues. Models will be loaded/unloaded as needed.
        self._available_models = self.enabled_models
        self._initialized = True
        logger.info(f"Multi-VLM initialization complete. Models will be loaded on-demand.")
    
    async def compare_models_on_element(
        self,
        visual_element: VisualElement,
        document_context: Optional[Dict[str, Any]] = None
    ) -> VLMComparison:
        """
        Compare all enabled VLM models on a single visual element
        
        Args:
            visual_element: Visual element to analyze
            document_context: Document context for analysis
            
        Returns:
            VLMComparison with results from all models
        """
        await self._initialize_clients()
        
        comparison = VLMComparison()
        
        # Generate adaptive prompt
        prompt_config = self.prompt_generator.generate_prompt(
            visual_element.element_type,
            document_context
        )
        
        # Analyze with each model sequentially to avoid GPU memory issues
        for model_type in self._available_models:
            client = None
            try:
                start_time = datetime.now()
                
                logger.info(f"Loading {model_type.value} for analysis...")
                
                # Initialize model based on type
                if model_type == VLMModelType.QWEN25_VL_7B:
                    try:
                        # Try vLLM first
                        client = VLLMQwen25VLClient(
                            gpu_memory_utilization=0.7,  # Use more GPU memory since only one model at a time
                            max_image_size=1024,
                            batch_size=2,
                            auto_load=True
                        )
                    except Exception as e:
                        logger.warning(f"vLLM failed for Qwen2.5-VL, falling back to Transformers: {e}")
                        # Fallback to Transformers
                        from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
                        client = TransformersQwen25VLClient(
                            temperature=0.1,
                            max_new_tokens=1024
                        )
                elif model_type == VLMModelType.LLAMA32_VISION_11B:
                    client = VLLMLlama32VisionClient(
                        gpu_memory_utilization=0.8,  # Llama needs more memory
                        max_image_size=1024,
                        batch_size=2,
                        auto_load=True
                    )
                elif model_type == VLMModelType.LLAVA_16_34B:
                    from core.clients.transformers_llava_client import TransformersLLaVAClient
                    client = TransformersLLaVAClient(
                        model_name="llava-hf/llava-v1.6-34b-hf",
                        load_in_4bit=True,  # Use 4-bit for 34B model
                        temperature=0.1,
                        max_new_tokens=1024
                    )
                elif model_type == VLMModelType.LLAVA_16_7B:
                    from core.clients.transformers_llava_client import TransformersLLaVAClient
                    client = TransformersLLaVAClient(
                        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
                        load_in_8bit=True,  # Use 8-bit for 7B model
                        temperature=0.1,
                        max_new_tokens=1024
                    )
                elif model_type == VLMModelType.PIXTRAL_12B:
                    from core.clients.transformers_pixtral_client import TransformersPixtralClient
                    client = TransformersPixtralClient(
                        temperature=0.1,
                        max_new_tokens=1024
                    )
                else:
                    logger.warning(f"⚠️ {model_type.value} not yet implemented")
                    continue
                
                logger.info(f"Analyzing element with {model_type.value}...")
                
                # Analyze with current model
                analysis_result = await asyncio.to_thread(
                    client.analyze_visual,
                    image_data=visual_element.raw_data,
                    document_context=prompt_config['context'],
                    element_type=visual_element.element_type,
                    analysis_focus=prompt_config['analysis_focus']
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Add result to comparison
                comparison.add_model_result(model_type.value, analysis_result, processing_time)
                
                logger.info(f"✅ {model_type.value}: {analysis_result.confidence:.2f} confidence")
                
            except Exception as e:
                logger.error(f"❌ {model_type.value} failed: {e}")
            finally:
                # Clean up model to free GPU memory
                if client:
                    try:
                        if hasattr(client, 'cleanup'):
                            client.cleanup()
                        del client
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {model_type.value}: {e}")
                    
                    # Give GPU time to free memory
                    await asyncio.sleep(2)
        
        # Calculate consensus
        consensus = comparison.calculate_consensus()
        logger.info(f"📊 Consensus: {consensus.get('consensus_confidence', 0):.2f} confidence, best: {consensus.get('best_model', 'none')}")
        
        return comparison
    
    async def analyze_visual_elements_comparative(
        self,
        visual_elements: List[VisualElement],
        document_context: Optional[Dict[str, Any]] = None,
        max_elements: int = 5  # Limit for performance
    ) -> Dict[str, VLMComparison]:
        """
        Perform comparative analysis on multiple visual elements
        
        Args:
            visual_elements: List of visual elements to analyze
            document_context: Document context for analysis
            max_elements: Maximum number of elements to analyze (for performance)
            
        Returns:
            Dict mapping element content_hash to VLMComparison
        """
        results = {}
        
        # Limit elements for performance
        elements_to_analyze = visual_elements[:max_elements]
        
        logger.info(f"🔍 Starting comparative analysis on {len(elements_to_analyze)} elements with {len(self.enabled_models)} models")
        
        for i, element in enumerate(elements_to_analyze, 1):
            try:
                if not element.raw_data:
                    logger.warning(f"No raw data for element {element.content_hash}")
                    continue
                
                logger.info(f"📈 Comparing element {i}/{len(elements_to_analyze)} ({element.element_type.value})")
                
                comparison = await self.compare_models_on_element(element, document_context)
                results[element.content_hash] = comparison
                
            except Exception as e:
                logger.error(f"Failed comparative analysis for element {element.content_hash}: {e}")
        
        logger.info(f"🎯 Comparative analysis complete: {len(results)} elements analyzed")
        return results


class VLMIntegration:
    """
    Handles VLM integration for visual element analysis
    """
    
    def __init__(self, vlm_client: Optional[VLLMQwen25VLClient] = None):
        """
        Initialize VLM integration
        
        Args:
            vlm_client: Optional pre-initialized VLM client
        """
        self.vlm_client = vlm_client
        self.prompt_generator = AdaptivePromptGenerator()
        self._initialized = False
        
    def _ensure_initialized(self):
        """Ensure VLM client is initialized"""
        if not self._initialized and self.vlm_client is None:
            logger.info("Initializing VLLMQwen25VLClient...")
            self.vlm_client = VLLMQwen25VLClient(
                gpu_memory_utilization=0.5,  # Conservative for multi-model setup
                max_image_size=1024,
                batch_size=4,
                auto_load=True
            )
            self._initialized = True
    
    async def analyze_visual_elements(
        self,
        visual_elements: List[VisualElement],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[VisualElement]:
        """
        Analyze visual elements with adaptive prompts
        
        Args:
            visual_elements: List of visual elements to analyze
            document_context: Document context for better analysis
            
        Returns:
            Updated visual elements with VLM descriptions
        """
        if not visual_elements:
            return visual_elements
        
        self._ensure_initialized()
        
        analyzed_elements = []
        
        for element in visual_elements:
            try:
                if not element.raw_data:
                    logger.warning(f"No raw data for element {element.content_hash}")
                    analyzed_elements.append(element)
                    continue
                
                # Generate adaptive prompt
                prompt_config = self.prompt_generator.generate_prompt(
                    element.element_type,
                    document_context
                )
                
                # Analyze with VLM
                logger.info(f"Analyzing {element.element_type.value} with focus: {prompt_config['analysis_focus']}")
                
                analysis_result = await asyncio.to_thread(
                    self.vlm_client.analyze_visual,
                    image_data=element.raw_data,
                    document_context=prompt_config['context'],
                    element_type=element.element_type,
                    analysis_focus=prompt_config['analysis_focus']
                )
                
                # Update element with results
                element.vlm_description = analysis_result.description
                element.confidence = analysis_result.confidence
                element.extracted_data = analysis_result.extracted_data
                
                # Merge analysis metadata
                if element.analysis_metadata is None:
                    element.analysis_metadata = {}
                element.analysis_metadata.update(analysis_result.metadata or {})
                element.analysis_metadata['analysis_focus'] = prompt_config['analysis_focus']
                element.analysis_metadata['analysis_timestamp'] = datetime.now().isoformat()
                
                analyzed_elements.append(element)
                logger.info(f"✅ Successfully analyzed {element.element_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to analyze element {element.content_hash}: {e}")
                analyzed_elements.append(element)
        
        return analyzed_elements
    
    async def update_visual_segments(
        self,
        segments: List[Segment],
        visual_elements: List[VisualElement]
    ) -> List[Segment]:
        """
        Update visual segments with VLM descriptions
        
        Args:
            segments: List of all segments
            visual_elements: List of analyzed visual elements
            
        Returns:
            Updated segments with visual descriptions
        """
        # Create lookup map for visual elements
        visual_map = {ve.content_hash: ve for ve in visual_elements}
        
        updated_segments = []
        
        for segment in segments:
            # Check if this is a visual segment
            if segment.segment_type == SegmentType.VISUAL:
                # Update content from placeholder to actual description
                if segment.visual_references:
                    hash_ref = segment.visual_references[0]  # Primary reference
                    
                    if hash_ref in visual_map:
                        visual_elem = visual_map[hash_ref]
                        
                        if visual_elem.vlm_description:
                            # Update content with VLM description
                            segment.content = visual_elem.vlm_description
                            
                            # Update metadata
                            if segment.metadata is None:
                                segment.metadata = {}
                            
                            segment.metadata.update({
                                "vlm_analyzed": True,
                                "confidence": visual_elem.confidence,
                                "has_extracted_data": bool(visual_elem.extracted_data),
                                "analysis_timestamp": visual_elem.analysis_metadata.get('analysis_timestamp')
                            })
                            
                            logger.info(f"Updated visual segment with VLM description")
                        else:
                            # Keep placeholder but mark as not analyzed
                            segment.metadata["vlm_analyzed"] = False
            
            updated_segments.append(segment)
        
        return updated_segments
    
    def cleanup(self):
        """Cleanup VLM client resources"""
        if self.vlm_client and self._initialized:
            try:
                # VLM client cleanup if needed
                logger.info("Cleaned up VLM client")
            except Exception as e:
                logger.error(f"Error during VLM cleanup: {e}")