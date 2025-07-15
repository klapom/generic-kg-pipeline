"""Generic Knowledge Graph Pipeline System - Core Module"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config_compat import Config, get_config, load_chunking_config

# Lazy imports to avoid circular dependencies
def get_batch_processor():
    from .batch_processor import BatchProcessor
    return BatchProcessor

def get_content_chunker():
    from .content_chunker import ContentChunker
    return ContentChunker

__all__ = [
    "Config", 
    "get_config", 
    "load_chunking_config",
    "get_batch_processor",
    "get_content_chunker",
    "__version__"
]