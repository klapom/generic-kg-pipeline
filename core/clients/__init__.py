"""Client modules for external services"""

from .vllm_smoldocling import VLLMSmolDoclingClient
from .hochschul_llm import HochschulLLMClient

__all__ = ["VLLMSmolDoclingClient", "HochschulLLMClient"]