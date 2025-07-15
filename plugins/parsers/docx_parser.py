"""
Compatibility stub for docx_parser.py
This module has been moved to core.parsers.implementations.office.docx_parser

This file is deprecated and will be removed in a future version.
Please update your imports to use the new location.
"""

import warnings
from core.parsers.implementations.office.docx_parser import *

warnings.warn(
    "Importing from 'plugins.parsers.docx_parser' is deprecated. "
    "Please use 'core.parsers.implementations.office.docx_parser' instead.",
    DeprecationWarning,
    stacklevel=2
)