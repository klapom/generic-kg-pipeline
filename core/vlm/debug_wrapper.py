"""
VLM Debug Wrapper for Pipeline Integration

Wraps VLM processors to provide debugging information and tracking.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from core.parsers import Document, VisualElement
from core.vlm.two_stage_processor import TwoStageVLMProcessor
from core.vlm.confidence_evaluator import ConfidenceEvaluator
from core.pipeline_debugger import PipelineDebugger


@dataclass
class VLMProcessingResult:
    """Result from VLM processing with debug info"""
    element_id: str
    element_type: str
    page_number: int
    model_used: str
    description: str
    confidence: float
    processing_time: float
    stage: int  # 1 or 2
    fallback_used: bool = False
    error: Optional[str] = None


class DebugVLMProcessor:
    """VLM processor with integrated debugging capabilities"""
    
    def __init__(self, debugger: Optional[PipelineDebugger] = None):
        self.debugger = debugger
        self.vlm_processor = TwoStageVLMProcessor()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.logger = logging.getLogger(__name__)
        self.results: List[VLMProcessingResult] = []
    
    async def process_document_visuals(self, document: Document) -> Document:
        """Process all visual elements in a document with debugging"""
        
        if not document.visual_elements:
            self.logger.info("No visual elements to process")
            return document
        
        self.logger.info(f"Processing {len(document.visual_elements)} visual elements")
        self.results.clear()
        
        # Classify elements for two-stage processing
        stage1_elements = []
        stage2_elements = []
        
        for visual in document.visual_elements:
            if visual.element_type in ["diagram", "chart", "flowchart"]:
                stage2_elements.append(visual)
            else:
                stage1_elements.append(visual)
        
        # Process Stage 1 (Qwen2.5-VL for general images)
        if stage1_elements:
            await self._process_stage1(document, stage1_elements)
        
        # Process Stage 2 (Pixtral for diagrams and fallbacks)
        if stage2_elements:
            await self._process_stage2(document, stage2_elements)
        
        # Process low-confidence results with fallback
        await self._process_fallbacks(document)
        
        # Update document segments with VLM descriptions
        self._update_document_segments(document)
        
        # Log summary
        self._log_processing_summary()
        
        return document
    
    async def _process_stage1(self, document: Document, elements: List[VisualElement]) -> None:
        """Process elements with Stage 1 VLM (Qwen2.5-VL)"""
        
        self.logger.info(f"Stage 1: Processing {len(elements)} elements with Qwen2.5-VL")
        
        for visual in elements:
            start_time = time.time()
            
            try:
                # Get analysis from VLM
                analysis = await self.vlm_processor.model_manager.analyze_with_qwen(
                    visual.image_data,
                    prompt=f"Describe this {visual.element_type} in detail."
                )
                
                processing_time = time.time() - start_time
                
                # Evaluate confidence
                confidence = self.confidence_evaluator.evaluate_single(analysis)
                
                # Create result
                result = VLMProcessingResult(
                    element_id=visual.element_id,
                    element_type=visual.element_type,
                    page_number=visual.page_number,
                    model_used="qwen2.5-vl",
                    description=analysis.description,
                    confidence=confidence.overall_score,
                    processing_time=processing_time,
                    stage=1
                )
                
                self.results.append(result)
                
                # Track in debugger if available
                if self.debugger:
                    self.debugger.track_vlm_processing(
                        segment_id=visual.element_id,
                        model="qwen2.5-vl",
                        description=analysis.description,
                        confidence=confidence.overall_score,
                        processing_time=processing_time
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing visual {visual.element_id}: {str(e)}")
                
                result = VLMProcessingResult(
                    element_id=visual.element_id,
                    element_type=visual.element_type,
                    page_number=visual.page_number,
                    model_used="qwen2.5-vl",
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    stage=1,
                    error=str(e)
                )
                
                self.results.append(result)
                
                if self.debugger:
                    self.debugger.track_error("vlm_stage1", e)
    
    async def _process_stage2(self, document: Document, elements: List[VisualElement]) -> None:
        """Process elements with Stage 2 VLM (Pixtral)"""
        
        self.logger.info(f"Stage 2: Processing {len(elements)} diagrams with Pixtral")
        
        # Clean up Stage 1 model first
        self.vlm_processor.model_manager.cleanup()
        
        for visual in elements:
            start_time = time.time()
            
            try:
                # Get analysis from Pixtral
                analysis = await self.vlm_processor.model_manager.analyze_with_pixtral(
                    visual.image_data,
                    prompt=f"Analyze this {visual.element_type} and describe its components and relationships."
                )
                
                processing_time = time.time() - start_time
                
                # Evaluate confidence
                confidence = self.confidence_evaluator.evaluate_single(analysis)
                
                # Create result
                result = VLMProcessingResult(
                    element_id=visual.element_id,
                    element_type=visual.element_type,
                    page_number=visual.page_number,
                    model_used="pixtral",
                    description=analysis.description,
                    confidence=confidence.overall_score,
                    processing_time=processing_time,
                    stage=2
                )
                
                self.results.append(result)
                
                # Track in debugger
                if self.debugger:
                    self.debugger.track_vlm_processing(
                        segment_id=visual.element_id,
                        model="pixtral",
                        description=analysis.description,
                        confidence=confidence.overall_score,
                        processing_time=processing_time
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing diagram {visual.element_id}: {str(e)}")
                
                result = VLMProcessingResult(
                    element_id=visual.element_id,
                    element_type=visual.element_type,
                    page_number=visual.page_number,
                    model_used="pixtral",
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    stage=2,
                    error=str(e)
                )
                
                self.results.append(result)
                
                if self.debugger:
                    self.debugger.track_error("vlm_stage2", e)
    
    async def _process_fallbacks(self, document: Document) -> None:
        """Process low-confidence results with fallback model"""
        
        # Find low-confidence results from Stage 1
        low_confidence_results = [
            r for r in self.results 
            if r.stage == 1 and r.confidence < 0.7 and not r.error
        ]
        
        if not low_confidence_results:
            return
        
        self.logger.info(f"Processing {len(low_confidence_results)} low-confidence results with Pixtral")
        
        # Make sure Pixtral is loaded
        if not hasattr(self.vlm_processor.model_manager, 'current_model_name') or \
           self.vlm_processor.model_manager.current_model_name != "pixtral":
            self.vlm_processor.model_manager.cleanup()
        
        for result in low_confidence_results:
            # Find visual element
            visual = next((v for v in document.visual_elements if v.element_id == result.element_id), None)
            if not visual:
                continue
            
            start_time = time.time()
            
            try:
                # Retry with Pixtral
                analysis = await self.vlm_processor.model_manager.analyze_with_pixtral(
                    visual.image_data,
                    prompt=f"Please provide a detailed analysis of this {visual.element_type}."
                )
                
                processing_time = time.time() - start_time
                
                # Evaluate new confidence
                confidence = self.confidence_evaluator.evaluate_single(analysis)
                
                # Update result if better
                if confidence.overall_score > result.confidence:
                    result.model_used = "pixtral (fallback)"
                    result.description = analysis.description
                    result.confidence = confidence.overall_score
                    result.processing_time += processing_time
                    result.fallback_used = True
                    
                    self.logger.info(f"Fallback improved confidence for {visual.element_id}: "
                                   f"{result.confidence:.2%}")
                    
                    # Track in debugger
                    if self.debugger:
                        self.debugger.track_vlm_processing(
                            segment_id=visual.element_id,
                            model="pixtral (fallback)",
                            description=analysis.description,
                            confidence=confidence.overall_score,
                            processing_time=processing_time
                        )
                
            except Exception as e:
                self.logger.error(f"Error in fallback processing for {visual.element_id}: {str(e)}")
                if self.debugger:
                    self.debugger.track_error("vlm_fallback", e)
    
    def _update_document_segments(self, document: Document) -> None:
        """Update document segments with VLM descriptions"""
        
        for result in self.results:
            if result.error or not result.description:
                continue
            
            # Find corresponding segment
            segment = next(
                (s for s in document.segments if s.visual_element_id == result.element_id),
                None
            )
            
            if segment:
                # Append VLM description to segment content
                vlm_text = f"\n\n[VLM Analysis ({result.model_used}, confidence: {result.confidence:.2%})]:\n{result.description}"
                segment.content += vlm_text
                
                # Add metadata
                if not hasattr(segment, 'metadata'):
                    segment.metadata = {}
                
                segment.metadata['vlm_analysis'] = {
                    'model': result.model_used,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'fallback_used': result.fallback_used
                }
    
    def _log_processing_summary(self) -> None:
        """Log summary of VLM processing"""
        
        total_elements = len(self.results)
        successful = sum(1 for r in self.results if not r.error)
        stage1_count = sum(1 for r in self.results if r.stage == 1)
        stage2_count = sum(1 for r in self.results if r.stage == 2)
        fallback_count = sum(1 for r in self.results if r.fallback_used)
        
        avg_confidence = sum(r.confidence for r in self.results if not r.error) / successful if successful > 0 else 0
        total_time = sum(r.processing_time for r in self.results)
        
        self.logger.info(f"""
VLM Processing Summary:
  Total elements: {total_elements}
  Successful: {successful}
  Failed: {total_elements - successful}
  Stage 1 (Qwen2.5-VL): {stage1_count}
  Stage 2 (Pixtral): {stage2_count}
  Fallbacks used: {fallback_count}
  Average confidence: {avg_confidence:.2%}
  Total processing time: {total_time:.2f}s
  Average time per element: {total_time/total_elements:.2f}s
        """)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        
        return {
            "total_elements": len(self.results),
            "successful": sum(1 for r in self.results if not r.error),
            "failed": sum(1 for r in self.results if r.error),
            "by_stage": {
                "stage1": sum(1 for r in self.results if r.stage == 1),
                "stage2": sum(1 for r in self.results if r.stage == 2)
            },
            "by_model": {
                "qwen2.5-vl": sum(1 for r in self.results if "qwen" in r.model_used),
                "pixtral": sum(1 for r in self.results if "pixtral" in r.model_used)
            },
            "fallbacks": sum(1 for r in self.results if r.fallback_used),
            "confidence": {
                "average": sum(r.confidence for r in self.results if not r.error) / len([r for r in self.results if not r.error]) if self.results else 0,
                "high": sum(1 for r in self.results if r.confidence >= 0.8),
                "medium": sum(1 for r in self.results if 0.5 <= r.confidence < 0.8),
                "low": sum(1 for r in self.results if r.confidence < 0.5)
            },
            "timing": {
                "total": sum(r.processing_time for r in self.results),
                "average": sum(r.processing_time for r in self.results) / len(self.results) if self.results else 0,
                "by_stage": {
                    "stage1": sum(r.processing_time for r in self.results if r.stage == 1),
                    "stage2": sum(r.processing_time for r in self.results if r.stage == 2)
                }
            },
            "errors": [
                {
                    "element_id": r.element_id,
                    "element_type": r.element_type,
                    "error": r.error
                }
                for r in self.results if r.error
            ]
        }