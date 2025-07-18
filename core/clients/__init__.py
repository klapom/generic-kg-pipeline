"""Client modules for external services"""

from .vllm_smoldocling_final import VLLMSmolDoclingFinalClient
from .hochschul_llm import HochschulLLMClient

__all__ = ["VLLMSmolDoclingFinalClient", "HochschulLLMClient"]