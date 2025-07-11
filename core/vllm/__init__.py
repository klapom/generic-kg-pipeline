"""
vLLM Integration Module

Provides local vLLM model management and inference capabilities
for high-performance document processing.
"""

from .model_manager import VLLMModelManager
from .base_client import BaseVLLMClient
from .exceptions import VLLMError, ModelNotLoadedError, GPUMemoryError

__all__ = [
    "VLLMModelManager",
    "BaseVLLMClient", 
    "VLLMError",
    "ModelNotLoadedError",
    "GPUMemoryError"
]