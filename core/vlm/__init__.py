"""
VLM (Visual Language Model) Pipeline Components

This module provides efficient VLM processing with:
- Model management for memory-efficient switching
- Document classification for intelligent routing  
- Two-stage processing (fast Qwen + precise Pixtral)
- Batch processing capabilities
- Confidence-based fallback mechanisms
"""

from .model_manager import VLMModelManager
from .document_classifier import DocumentElementClassifier
from .two_stage_processor import TwoStageVLMProcessor
from .batch_processor import BatchDocumentProcessor, BatchProcessingConfig, BatchResult
from .confidence_evaluator import ConfidenceEvaluator, FallbackStrategy, FallbackReason

__all__ = [
    "VLMModelManager",
    "DocumentElementClassifier", 
    "TwoStageVLMProcessor",
    "BatchDocumentProcessor",
    "BatchProcessingConfig",
    "BatchResult",
    "ConfidenceEvaluator",
    "FallbackStrategy",
    "FallbackReason"
]