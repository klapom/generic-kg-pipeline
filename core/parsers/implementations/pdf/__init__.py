"""
PDF parsing implementations

New Architecture (Recommended):
- PDFProcessor: Main orchestrator using PDFPreprocessor + BasePDFParser + ImageAnalyzer
- BasePDFParser: SmolDocling-based text/structure extraction
- ImageAnalyzer: Qwen2.5-VL analysis for embedded images and pages
- PDFPreprocessor: Unified image extraction and caching

Legacy (For backward compatibility):
- ImageExtractionPDFParser: Specialized parser for image extraction with Qwen2.5-VL
- HybridPDFParserQwen25: Enhanced parser with Qwen2.5-VL, table/chart detection
- HybridPDFParser: Complete parser with SmolDocling + fallback + VLM support
- PDFParser: Multi-modal parser with VLM support (legacy)
"""

# New architecture imports
try:
    from .pdf_processor import PDFProcessor
    from .base_pdf_parser import BasePDFParser
    from .image_analyzer import ImageAnalyzer
    from .pdf_preprocessor import PDFPreprocessor
    NEW_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"New PDF architecture not available: {e}")
    NEW_ARCHITECTURE_AVAILABLE = False
    # Create placeholder classes
    PDFProcessor = None
    BasePDFParser = None
    ImageAnalyzer = None
    PDFPreprocessor = None

# Legacy imports
from .standard_pdf_parser import PDFParser
from .hybrid_pdf_parser import HybridPDFParser
from .hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25
from .image_extraction_parser import ImageExtractionPDFParser

__all__ = [
    # New architecture
    'PDFProcessor',
    'BasePDFParser', 
    'ImageAnalyzer',
    'PDFPreprocessor',
    # Legacy
    'ImageExtractionPDFParser',
    'HybridPDFParserQwen25',
    'HybridPDFParser',
    'PDFParser'
]