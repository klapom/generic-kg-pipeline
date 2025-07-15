"""
Configuration module - Bridge to compatibility layer
This file exists to maintain backward compatibility with imports like:
from core.config import Config, get_config
"""

# Import everything from compatibility layer
from .config_compat import *

# Make sure all exports are available
__all__ = [
    'Config',
    'get_config', 
    'load_chunking_config',
    'PDFParsingConfig',
    'LLMConfig',
    'ChunkingConfig',
    'StorageConfig',
    'DomainConfig',
    'VLLMConfig',
    'BatchProcessingConfig',
    'HochschulLLMConfig',
    'OllamaConfig',
    'ParsingConfig',
    'TripleStoreConfig',
    'VectorStoreConfig',
    'RAGConfig'
]