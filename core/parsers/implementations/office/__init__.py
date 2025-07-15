"""
Office parser implementations (DOCX, XLSX, PPTX)
"""

from .docx_parser import DOCXParser
from .xlsx_parser import XLSXParser
from .pptx_parser import PPTXParser

__all__ = ['DOCXParser', 'XLSXParser', 'PPTXParser']