#!/usr/bin/env python3
"""
Confidence Evaluator and Fallback Strategy for VLM results.
Implements intelligent decision-making based on confidence scores and result quality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re

from core.parsers.interfaces.data_models import VisualAnalysisResult, VisualElementType

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for triggering fallback processing."""
    LOW_CONFIDENCE = "low_confidence"
    PARSING_ERROR = "parsing_error"
    INCOMPLETE_DATA = "incomplete_data"
    HALLUCINATION_DETECTED = "hallucination_detected"
    CRITICAL_DOCUMENT = "critical_document"
    DIAGRAM_TYPE = "diagram_type"


@dataclass
class ConfidenceMetrics:
    """Detailed confidence metrics for a result."""
    overall_confidence: float
    text_quality: float
    data_completeness: float
    hallucination_score: float
    needs_fallback: bool
    fallback_reasons: List[FallbackReason]


class ConfidenceEvaluator:
    """
    Evaluates VLM results and determines if fallback processing is needed.
    """
    
    def __init__(self, 
                 base_threshold: float = 0.85,
                 diagram_threshold: float = 0.90,
                 critical_threshold: float = 0.95):
        """
        Initialize the confidence evaluator.
        
        Args:
            base_threshold: Default confidence threshold
            diagram_threshold: Higher threshold for diagrams
            critical_threshold: Threshold for critical documents
        """
        self.base_threshold = base_threshold
        self.diagram_threshold = diagram_threshold
        self.critical_threshold = critical_threshold
        
        # Hallucination patterns
        self.generic_patterns = [
            r"image of a \w+",
            r"picture showing",
            r"photo of",
            r"illustration of",
            r"generic \w+",
            r"various \w+",
            r"some kind of",
            r"appears to be",
            r"possibly a"
        ]
        
        # Quality indicators
        self.quality_indicators = {
            "specific_terms": ["diagram", "table", "chart", "graph", "BMW", "specifications"],
            "data_markers": ["values", "numbers", "measurements", "data", "statistics"],
            "structure_markers": ["rows", "columns", "cells", "headers", "labels"]
        }
    
    def evaluate(self, 
                result: VisualAnalysisResult,
                element_type: Optional[VisualElementType] = None,
                is_critical: bool = False) -> ConfidenceMetrics:
        """
        Comprehensive evaluation of a VLM result.
        
        Args:
            result: VLM analysis result
            element_type: Type of visual element
            is_critical: Whether this is a critical document
            
        Returns:
            Detailed confidence metrics
        """
        if not result.success:
            return ConfidenceMetrics(
                overall_confidence=0.0,
                text_quality=0.0,
                data_completeness=0.0,
                hallucination_score=1.0,
                needs_fallback=True,
                fallback_reasons=[FallbackReason.PARSING_ERROR]
            )
        
        # Calculate individual metrics
        text_quality = self._evaluate_text_quality(result)
        data_completeness = self._evaluate_data_completeness(result)
        hallucination_score = self._detect_hallucination(result)
        
        # Determine appropriate threshold
        threshold = self._get_threshold(element_type, is_critical)
        
        # Calculate overall confidence (weighted average)
        overall_confidence = (
            result.confidence * 0.4 +
            text_quality * 0.3 +
            data_completeness * 0.2 +
            (1 - hallucination_score) * 0.1
        )
        
        # Determine if fallback is needed
        fallback_reasons = []
        
        if overall_confidence < threshold:
            fallback_reasons.append(FallbackReason.LOW_CONFIDENCE)
        
        if hallucination_score > 0.5:
            fallback_reasons.append(FallbackReason.HALLUCINATION_DETECTED)
        
        if data_completeness < 0.5 and element_type in [VisualElementType.TABLE, VisualElementType.DIAGRAM]:
            fallback_reasons.append(FallbackReason.INCOMPLETE_DATA)
        
        if is_critical and overall_confidence < self.critical_threshold:
            fallback_reasons.append(FallbackReason.CRITICAL_DOCUMENT)
        
        if element_type == VisualElementType.DIAGRAM and not self._is_diagram_well_analyzed(result):
            fallback_reasons.append(FallbackReason.DIAGRAM_TYPE)
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            text_quality=text_quality,
            data_completeness=data_completeness,
            hallucination_score=hallucination_score,
            needs_fallback=len(fallback_reasons) > 0,
            fallback_reasons=fallback_reasons
        )
    
    def _get_threshold(self, element_type: Optional[VisualElementType], is_critical: bool) -> float:
        """Get appropriate confidence threshold."""
        if is_critical:
            return self.critical_threshold
        elif element_type == VisualElementType.DIAGRAM:
            return self.diagram_threshold
        else:
            return self.base_threshold
    
    def _evaluate_text_quality(self, result: VisualAnalysisResult) -> float:
        """
        Evaluate the quality of text in the result.
        
        Returns:
            Score between 0 and 1
        """
        if not result.description:
            return 0.0
        
        score = 0.0
        description_lower = result.description.lower()
        
        # Check for specific terms (good)
        specific_count = sum(1 for term in self.quality_indicators["specific_terms"] 
                           if term in description_lower)
        score += min(specific_count * 0.1, 0.3)
        
        # Check description length (moderate length is good)
        desc_length = len(result.description.split())
        if 20 <= desc_length <= 200:
            score += 0.3
        elif 10 <= desc_length < 20:
            score += 0.2
        elif desc_length > 200:
            score += 0.2  # Still decent but might be verbose
        
        # Check for structure markers if applicable
        if any(marker in description_lower for marker in self.quality_indicators["structure_markers"]):
            score += 0.2
        
        # Check OCR text quality
        if result.ocr_text and len(result.ocr_text) > 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_data_completeness(self, result: VisualAnalysisResult) -> float:
        """
        Evaluate how complete the extracted data is.
        
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Check if extracted_data exists and has content
        if result.extracted_data:
            if isinstance(result.extracted_data, dict):
                # More keys = more complete
                key_count = len(result.extracted_data)
                score += min(key_count * 0.1, 0.5)
                
                # Check for nested data (good sign)
                has_nested = any(isinstance(v, (dict, list)) for v in result.extracted_data.values())
                if has_nested:
                    score += 0.3
                
                # Check for numeric data
                has_numbers = any(isinstance(v, (int, float)) or 
                                (isinstance(v, str) and any(c.isdigit() for c in v))
                                for v in str(result.extracted_data.values()))
                if has_numbers:
                    score += 0.2
            
            elif isinstance(result.extracted_data, list) and len(result.extracted_data) > 0:
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_hallucination(self, result: VisualAnalysisResult) -> float:
        """
        Detect potential hallucinations in the result.
        
        Returns:
            Hallucination score between 0 (no hallucination) and 1 (definite hallucination)
        """
        if not result.description:
            return 0.5  # Unknown
        
        description_lower = result.description.lower()
        hallucination_score = 0.0
        
        # Check for generic patterns
        for pattern in self.generic_patterns:
            if re.search(pattern, description_lower):
                hallucination_score += 0.2
        
        # Check for vagueness
        vague_terms = ["something", "stuff", "things", "various", "some", "certain"]
        vague_count = sum(1 for term in vague_terms if term in description_lower)
        hallucination_score += vague_count * 0.1
        
        # Check for contradictions with OCR
        if result.ocr_text and len(result.ocr_text) > 20:
            # If we have good OCR but generic description, likely hallucination
            if hallucination_score > 0.3:
                hallucination_score += 0.2
        
        return min(hallucination_score, 1.0)
    
    def _is_diagram_well_analyzed(self, result: VisualAnalysisResult) -> bool:
        """
        Check if a diagram has been properly analyzed.
        
        Returns:
            True if diagram analysis seems complete
        """
        if not result.description:
            return False
        
        description_lower = result.description.lower()
        
        # Diagram-specific terms
        diagram_terms = [
            "flow", "connection", "relationship", "arrow", "box", "node",
            "process", "step", "component", "system", "architecture",
            "input", "output", "link", "branch"
        ]
        
        # Count diagram-specific terms
        term_count = sum(1 for term in diagram_terms if term in description_lower)
        
        # Check for relationship descriptions
        has_relationships = any(word in description_lower for word in 
                              ["connects", "links", "flows", "leads", "between", "from", "to"])
        
        return term_count >= 2 and has_relationships


class FallbackStrategy:
    """
    Implements various fallback strategies based on confidence evaluation.
    """
    
    def __init__(self, evaluator: Optional[ConfidenceEvaluator] = None):
        """
        Initialize the fallback strategy.
        
        Args:
            evaluator: Confidence evaluator to use
        """
        self.evaluator = evaluator or ConfidenceEvaluator()
    
    def should_fallback(self,
                       result: VisualAnalysisResult,
                       element_type: Optional[VisualElementType] = None,
                       is_critical: bool = False) -> Tuple[bool, List[FallbackReason]]:
        """
        Determine if fallback processing is needed.
        
        Args:
            result: VLM analysis result
            element_type: Type of visual element
            is_critical: Whether this is a critical document
            
        Returns:
            Tuple of (needs_fallback, reasons)
        """
        metrics = self.evaluator.evaluate(result, element_type, is_critical)
        return metrics.needs_fallback, metrics.fallback_reasons
    
    def get_fallback_config(self, reasons: List[FallbackReason]) -> Dict[str, Any]:
        """
        Get optimized configuration for fallback processing.
        
        Args:
            reasons: Reasons for fallback
            
        Returns:
            Configuration dict for fallback processing
        """
        config = {
            "temperature": 0.3,  # Default
            "max_new_tokens": 512,
            "analysis_focus": "comprehensive"
        }
        
        # Adjust based on reasons
        if FallbackReason.HALLUCINATION_DETECTED in reasons:
            config["temperature"] = 0.1  # Lower temperature for less creativity
            config["analysis_focus"] = "precise_extraction"
        
        if FallbackReason.INCOMPLETE_DATA in reasons:
            config["max_new_tokens"] = 1024  # Allow more tokens
            config["analysis_focus"] = "detailed_analysis"
        
        if FallbackReason.DIAGRAM_TYPE in reasons:
            config["analysis_focus"] = "diagram_analysis"
            config["temperature"] = 0.2
        
        if FallbackReason.CRITICAL_DOCUMENT in reasons:
            config["temperature"] = 0.1
            config["max_new_tokens"] = 1024
            config["analysis_focus"] = "exhaustive_analysis"
        
        return config
    
    def merge_results(self,
                     primary_result: VisualAnalysisResult,
                     fallback_result: VisualAnalysisResult,
                     strategy: str = "replace") -> VisualAnalysisResult:
        """
        Merge primary and fallback results.
        
        Args:
            primary_result: Original result
            fallback_result: Fallback result
            strategy: Merge strategy ('replace', 'combine', 'best')
            
        Returns:
            Merged result
        """
        if strategy == "replace":
            # Simply use fallback result
            return fallback_result
        
        elif strategy == "combine":
            # Combine information from both
            combined_description = f"{primary_result.description}\n\nAdditional details: {fallback_result.description}"
            combined_ocr = (primary_result.ocr_text or "") + "\n" + (fallback_result.ocr_text or "")
            
            # Merge extracted data
            combined_data = {}
            if isinstance(primary_result.extracted_data, dict):
                combined_data.update(primary_result.extracted_data)
            if isinstance(fallback_result.extracted_data, dict):
                combined_data.update(fallback_result.extracted_data)
            
            return VisualAnalysisResult(
                success=True,
                confidence=max(primary_result.confidence, fallback_result.confidence),
                description=combined_description,
                ocr_text=combined_ocr.strip(),
                extracted_data=combined_data if combined_data else None
            )
        
        elif strategy == "best":
            # Choose the better result based on confidence
            primary_metrics = self.evaluator.evaluate(primary_result)
            fallback_metrics = self.evaluator.evaluate(fallback_result)
            
            if fallback_metrics.overall_confidence > primary_metrics.overall_confidence:
                return fallback_result
            else:
                return primary_result
        
        else:
            logger.warning(f"Unknown merge strategy: {strategy}, using 'replace'")
            return fallback_result