"""
Table and text separation utilities for hybrid parsing

DEPRECATED: This module is deprecated. Import from new location:
- from core.parsers.strategies.table_text_separator import TableTextSeparator, ContentRegion, clean_page_content
"""

import logging

# Compatibility imports
from core.parsers.strategies.table_text_separator import (
    TableTextSeparator,
    ContentRegion,
    clean_page_content
)

logger = logging.getLogger(__name__)
logger.warning("core.parsers.table_text_separator is deprecated. Import from core.parsers.strategies.table_text_separator instead.")

# Re-export for backward compatibility
__all__ = ['TableTextSeparator', 'ContentRegion', 'clean_page_content']