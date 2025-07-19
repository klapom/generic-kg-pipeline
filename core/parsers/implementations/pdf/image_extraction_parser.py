"""
Image Extraction PDF Parser

A specialized PDF parser focused solely on extracting and analyzing images
using Qwen2.5-VL. This is a wrapper around HybridPDFParserQwen25 with
pre-configured settings for image-only processing.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25
from core.parsers.interfaces import Document, DocumentType

logger = logging.getLogger(__name__)


class ImageExtractionPDFParser(HybridPDFParserQwen25):
    """
    PDF Parser specialized for image extraction and VLM analysis
    
    Features:
    - Extracts all embedded images from PDFs
    - Analyzes images with Qwen2.5-VL
    - Disables table/chart content detection
    - Optimized for visual element processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize image extraction parser with optimized settings
        
        Args:
            config: Optional configuration overrides
        """
        # Create base config for image extraction
        image_config = {
            # Disable content detection (tables/charts)
            'enable_content_detection': False,
            
            # Enable all image extraction features
            'extract_images': True,
            'extract_tables': False,  # Don't extract tables as visual elements
            'extract_formulas': True,  # Formulas are visual
            'image_min_size': 50,     # Smaller threshold for images
            
            # VLM configuration
            'vlm': {
                'temperature': 0.2,
                'max_new_tokens': 512,
                'batch_size': 4,
                'enable_structured_parsing': True,
                'enable_page_context': False  # Disable for performance
            },
            
            # Image extraction strategy
            'image_extraction': {
                'min_size': 50,
                'extract_embedded': True,
                'render_fallback': True,
                'page_render_dpi': 150,
                'render_visual_elements': True
            },
            
            # Performance settings
            'max_pages': 50,
            'pdfplumber_mode': 1,  # Fallback only
            'enable_page_context': False,  # Disable for faster processing
            
            # Environment
            'environment': 'production'
        }
        
        # Merge with user config if provided
        if config:
            # Deep merge vlm and image_extraction configs
            if 'vlm' in config:
                image_config['vlm'].update(config['vlm'])
            if 'image_extraction' in config:
                image_config['image_extraction'].update(config['image_extraction'])
            
            # Update other top-level configs
            for key in ['max_pages', 'environment', 'image_min_size']:
                if key in config:
                    image_config[key] = config[key]
        
        # Initialize parent with image-focused config
        super().__init__(config=image_config, enable_vlm=True)
        
        logger.info("Initialized ImageExtractionPDFParser - focused on visual element extraction")
    
    async def parse(self, file_path: Path) -> Document:
        """
        Parse PDF focusing on image extraction
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document with extracted and analyzed visual elements
        """
        logger.info(f"🖼️ Starting image-focused parsing of: {file_path.name}")
        
        # Use parent's parse method
        document = await super().parse(file_path)
        
        # Log extraction results
        logger.info(f"✅ Image extraction completed:")
        logger.info(f"   - Visual elements found: {len(document.visual_elements)}")
        
        # Count analyzed elements
        analyzed = sum(1 for ve in document.visual_elements if ve.vlm_description)
        logger.info(f"   - Elements analyzed by VLM: {analyzed}")
        
        # Log types of visual elements found
        element_types = {}
        for ve in document.visual_elements:
            element_type = ve.element_type.value
            element_types[element_type] = element_types.get(element_type, 0) + 1
        
        if element_types:
            logger.info("   - Element types:")
            for elem_type, count in element_types.items():
                logger.info(f"     • {elem_type}: {count}")
        
        return document
    
    def get_info(self) -> Dict[str, Any]:
        """Get parser information"""
        return {
            "parser": "ImageExtractionPDFParser",
            "version": "1.0",
            "features": [
                "Embedded image extraction",
                "Qwen2.5-VL image analysis",
                "Formula detection",
                "Page rendering fallback"
            ],
            "vlm_enabled": self.enable_vlm,
            "content_detection_enabled": self.enable_content_detection,
            "supported_formats": list(self.supported_types)
        }