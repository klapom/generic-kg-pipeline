"""Client modules for external services"""

from .vllm_smoldocling_final import VLLMSmolDoclingFinalClient
from .hochschul_llm import HochschulLLMClient

# Legacy compatibility alias
VLLMSmolDoclingClient = VLLMSmolDoclingFinalClient

__all__ = ["VLLMSmolDoclingClient", "VLLMSmolDoclingFinalClient", "HochschulLLMClient"]