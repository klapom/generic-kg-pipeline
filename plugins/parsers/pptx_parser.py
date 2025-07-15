"""
Compatibility stub for pptx_parser.py
This module has been moved to core.parsers.implementations.office.pptx_parser

This file is deprecated and will be removed in a future version.
Please update your imports to use the new location.
"""

import warnings
from core.parsers.implementations.office.pptx_parser import *

warnings.warn(
    "Importing from 'plugins.parsers.pptx_parser' is deprecated. "
    "Please use 'core.parsers.implementations.office.pptx_parser' instead.",
    DeprecationWarning,
    stacklevel=2
)