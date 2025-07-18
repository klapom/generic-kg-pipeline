"""
VLM (Visual Language Model) Pipeline Components

This module provides efficient VLM processing with:
- Model management for memory-efficient switching
- Document classification for intelligent routing  
- Two-stage processing (fast Qwen + precise Pixtral)
- Single-stage Qwen2.5-VL processor (NEW)
- Batch processing capabilities
- Confidence-based fallback mechanisms
- Enhanced image extraction strategies (NEW)
"""

from .model_manager import VLMModelManager
from .document_classifier import DocumentElementClassifier
from .two_stage_processor import TwoStageVLMProcessor
from .batch_processor import BatchDocumentProcessor, BatchProcessingConfig, BatchResult
from .confidence_evaluator import ConfidenceEvaluator, FallbackStrategy, FallbackReason

# New Qwen2.5-VL components
from .qwen25_processor import (
    Qwen25VLMProcessor,
    VisualAnalysisResult,
    PageContext
)

from .image_extraction import (
    ImageExtractionStrategy,
    ImageData
)

__all__ = [
    "VLMModelManager",
    "DocumentElementClassifier", 
    "TwoStageVLMProcessor",
    "BatchDocumentProcessor",
    "BatchProcessingConfig",
    "BatchResult",
    "ConfidenceEvaluator",
    "FallbackStrategy",
    "FallbackReason",
    # New exports
    "Qwen25VLMProcessor",
    "VisualAnalysisResult",
    "PageContext",
    "ImageExtractionStrategy",
    "ImageData"
]