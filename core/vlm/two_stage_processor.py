#!/usr/bin/env python3
"""
Two-Stage VLM Processor for efficient document analysis.
Stage 1: Fast processing with Qwen2.5-VL
Stage 2: High-precision processing with Pixtral for diagrams
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio

from core.vlm.model_manager import VLMModelManager
from core.vlm.document_classifier import DocumentElementClassifier
from core.vlm.confidence_evaluator import ConfidenceEvaluator, FallbackStrategy, FallbackReason
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, VisualAnalysisResult

logger = logging.getLogger(__name__)


class TwoStageVLMProcessor:
    """
    Orchestrates two-stage processing of documents using different VLMs.
    Optimizes for speed and accuracy by routing images to appropriate models.
    """
    
    def __init__(self, confidence_threshold: float = 0.85):
        """
        Initialize the two-stage processor.
        
        Args:
            confidence_threshold: Minimum confidence for accepting results
        """
        self.model_manager = VLMModelManager()
        self.classifier = DocumentElementClassifier()
        self.confidence_threshold = confidence_threshold
        self.confidence_evaluator = ConfidenceEvaluator(base_threshold=confidence_threshold)
        self.fallback_strategy = FallbackStrategy(self.confidence_evaluator)
        self.processing_stats = {
            "total_images": 0,
            "qwen_processed": 0,
            "pixtral_processed": 0,
            "fallback_count": 0,
            "total_time": 0
        }
    
    def process_batch(self, visual_elements: List[VisualElement]) -> List[VisualAnalysisResult]:
        """
        Process a batch of visual elements using two-stage strategy.
        
        Args:
            visual_elements: List of visual elements to process
            
        Returns:
            List of analysis results
        """
        start_time = time.time()
        self.processing_stats["total_images"] = len(visual_elements)
        
        logger.info(f"Starting two-stage processing for {len(visual_elements)} images")
        
        # Classify all images
        classified_elements = self._classify_elements(visual_elements)
        
        # Separate by processing needs
        qwen_batch = []
        pixtral_batch = []
        
        for element, classification in classified_elements:
            if classification["primary_model"] == "qwen":
                qwen_batch.append((element, classification))
            else:
                pixtral_batch.append((element, classification))
        
        logger.info(f"Routing: {len(qwen_batch)} to Qwen, {len(pixtral_batch)} to Pixtral")
        
        # Stage 1: Process with Qwen
        qwen_results = []
        if qwen_batch:
            qwen_results = self._stage1_qwen_processing(qwen_batch)
        
        # Stage 2: Process with Pixtral (includes diagrams and fallbacks)
        pixtral_results = []
        if pixtral_batch or self._get_fallback_elements(qwen_results):
            # Add fallback elements from Qwen
            fallback_elements = self._get_fallback_elements(qwen_results)
            pixtral_batch.extend(fallback_elements)
            pixtral_results = self._stage2_pixtral_processing(pixtral_batch)
        
        # Merge results
        all_results = self._merge_results(qwen_results, pixtral_results)
        
        # Update stats
        self.processing_stats["total_time"] = time.time() - start_time
        self._log_stats()
        
        return all_results
    
    def _classify_elements(self, visual_elements: List[VisualElement]) -> List[Tuple[VisualElement, Dict[str, Any]]]:
        """
        Classify all visual elements for routing.
        
        Args:
            visual_elements: List of visual elements
            
        Returns:
            List of (element, classification) tuples
        """
        classified = []
        
        for element in visual_elements:
            classification = self.classifier.get_processing_recommendation(
                element.raw_data,
                self.confidence_threshold
            )
            classified.append((element, classification))
            
        return classified
    
    def _stage1_qwen_processing(self, elements: List[Tuple[VisualElement, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Stage 1: Fast processing with Qwen2.5-VL.
        
        Args:
            elements: List of (element, classification) tuples
            
        Returns:
            List of results with metadata
        """
        logger.info(f"Stage 1: Processing {len(elements)} images with Qwen2.5-VL")
        
        # Load Qwen model
        qwen_model = self.model_manager.load_qwen()
        results = []
        
        for element, classification in elements:
            try:
                start = time.time()
                
                # Process with hints from classification
                result = qwen_model.analyze_visual(
                    image_data=element.raw_data,
                    element_type=element.element_type,
                    analysis_focus="comprehensive"
                )
                
                elapsed = time.time() - start
                
                # Evaluate result with confidence evaluator
                needs_fallback, fallback_reasons = self.fallback_strategy.should_fallback(
                    result, 
                    element.element_type,
                    classification.get("is_critical", False)
                )
                
                # Store result with metadata
                results.append({
                    "element": element,
                    "classification": classification,
                    "result": result,
                    "model": "qwen",
                    "processing_time": elapsed,
                    "needs_fallback": needs_fallback,
                    "fallback_reasons": fallback_reasons
                })
                
                self.processing_stats["qwen_processed"] += 1
                
                if result.success:
                    logger.info(f"✅ Qwen processed {element.content_hash}: "
                              f"{result.confidence:.0%} confidence in {elapsed:.1f}s")
                else:
                    logger.warning(f"❌ Qwen failed for {element.content_hash}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error processing {element.content_hash} with Qwen: {e}")
                results.append({
                    "element": element,
                    "classification": classification,
                    "result": VisualAnalysisResult(
                        success=False,
                        error_message=str(e),
                        confidence=0.0
                    ),
                    "model": "qwen",
                    "processing_time": time.time() - start,
                    "needs_fallback": True
                })
        
        return results
    
    def _stage2_pixtral_processing(self, elements: List[Tuple[VisualElement, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Stage 2: High-precision processing with Pixtral.
        
        Args:
            elements: List of (element, classification) tuples
            
        Returns:
            List of results with metadata
        """
        logger.info(f"Stage 2: Processing {len(elements)} images with Pixtral")
        
        # Clean up Qwen and load Pixtral
        self.model_manager.cleanup()
        pixtral_model = self.model_manager.load_pixtral()
        results = []
        
        for item in elements:
            # Handle both original elements and fallback elements
            if isinstance(item, tuple):
                element, classification = item
                is_fallback = False
            else:  # Fallback element
                element = item["element"]
                classification = item["classification"]
                is_fallback = True
            
            try:
                start = time.time()
                
                # Get optimized config for fallback if needed
                if is_fallback and "fallback_reasons" in item:
                    fallback_config = self.fallback_strategy.get_fallback_config(item["fallback_reasons"])
                    analysis_focus = fallback_config.get("analysis_focus", "comprehensive")
                else:
                    # Process with specific focus for diagrams
                    analysis_focus = "diagram_analysis" if classification["element_type"] == VisualElementType.DIAGRAM else "comprehensive"
                
                result = pixtral_model.analyze_visual(
                    image_data=element.raw_data,
                    element_type=element.element_type,
                    analysis_focus=analysis_focus
                )
                
                elapsed = time.time() - start
                
                # Store result with metadata
                results.append({
                    "element": element,
                    "classification": classification,
                    "result": result,
                    "model": "pixtral",
                    "processing_time": elapsed,
                    "is_fallback": is_fallback
                })
                
                self.processing_stats["pixtral_processed"] += 1
                if is_fallback:
                    self.processing_stats["fallback_count"] += 1
                
                if result.success:
                    logger.info(f"✅ Pixtral processed {element.content_hash}: "
                              f"{result.confidence:.0%} confidence in {elapsed:.1f}s"
                              f"{' (fallback)' if is_fallback else ''}")
                else:
                    logger.warning(f"❌ Pixtral failed for {element.content_hash}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error processing {element.content_hash} with Pixtral: {e}")
                results.append({
                    "element": element,
                    "classification": classification,
                    "result": VisualAnalysisResult(
                        success=False,
                        error_message=str(e),
                        confidence=0.0
                    ),
                    "model": "pixtral",
                    "processing_time": time.time() - start,
                    "is_fallback": is_fallback
                })
        
        return results
    
    def _get_fallback_elements(self, qwen_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get elements that need fallback processing.
        
        Args:
            qwen_results: Results from Qwen processing
            
        Returns:
            Elements needing fallback
        """
        fallback_elements = []
        
        for result in qwen_results:
            if result.get("needs_fallback", False):
                fallback_elements.append(result)
                logger.info(f"Element {result['element'].content_hash} needs fallback "
                          f"(confidence: {result['result'].confidence:.0%})")
        
        return fallback_elements
    
    def _merge_results(self, qwen_results: List[Dict[str, Any]], 
                      pixtral_results: List[Dict[str, Any]]) -> List[VisualAnalysisResult]:
        """
        Merge results from both stages, preferring Pixtral for fallbacks.
        
        Args:
            qwen_results: Results from Qwen processing
            pixtral_results: Results from Pixtral processing
            
        Returns:
            Final list of VisualAnalysisResult objects
        """
        # Create lookup for Pixtral results
        pixtral_lookup = {
            result["element"].content_hash: result 
            for result in pixtral_results
        }
        
        final_results = []
        
        # Process Qwen results
        for qwen_result in qwen_results:
            element_id = qwen_result["element"].content_hash
            
            # Check if we have a Pixtral fallback
            if element_id in pixtral_lookup:
                # Use Pixtral result
                final_results.append(pixtral_lookup[element_id]["result"])
                logger.info(f"Using Pixtral result for {element_id} (fallback)")
            else:
                # Use Qwen result
                final_results.append(qwen_result["result"])
        
        # Add Pixtral-only results (original diagrams)
        for pixtral_result in pixtral_results:
            if not pixtral_result.get("is_fallback", False):
                final_results.append(pixtral_result["result"])
        
        return final_results
    
    def _log_stats(self):
        """Log processing statistics."""
        stats = self.processing_stats
        logger.info(f"""
Two-Stage Processing Complete:
- Total images: {stats['total_images']}
- Qwen processed: {stats['qwen_processed']}
- Pixtral processed: {stats['pixtral_processed']}
- Fallback count: {stats['fallback_count']}
- Total time: {stats['total_time']:.1f}s
- Average time per image: {stats['total_time'] / max(stats['total_images'], 1):.1f}s
- Memory usage: {self.model_manager.get_memory_usage()}
""")
    
    def cleanup(self):
        """Clean up resources."""
        self.model_manager.cleanup()