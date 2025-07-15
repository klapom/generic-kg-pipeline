"""
Text parser compatibility layer

DEPRECATED: This module provides backward compatibility. 
Use the new parser from:
- from core.parsers.implementations.text import TXTParser
"""

import logging

# Compatibility import
from core.parsers.implementations.text import TXTParser

logger = logging.getLogger(__name__)
logger.warning("plugins.parsers.txt_parser is deprecated. Import from core.parsers.implementations.text instead.")

# Re-export for backward compatibility
__all__ = ['TXTParser']