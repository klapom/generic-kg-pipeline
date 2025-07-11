"""
vLLM-specific exceptions
"""


class VLLMError(Exception):
    """Base exception for vLLM-related errors"""
    pass


class ModelNotLoadedError(VLLMError):
    """Raised when trying to use a model that hasn't been loaded"""
    pass


class GPUMemoryError(VLLMError):
    """Raised when there's insufficient GPU memory"""
    pass


class ModelLoadError(VLLMError):
    """Raised when model loading fails"""
    pass


class InferenceError(VLLMError):
    """Raised when inference fails"""
    pass


class ConfigurationError(VLLMError):
    """Raised when vLLM configuration is invalid"""
    pass