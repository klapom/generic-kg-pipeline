"""Generic Knowledge Graph Pipeline System - Core Module"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Config, get_config, load_chunking_config
from .batch_processor import BatchProcessor
from .content_chunker import ContentChunker

__all__ = [
    "Config", 
    "get_config", 
    "load_chunking_config",
    "BatchProcessor",
    "ContentChunker",
    "__version__"
]