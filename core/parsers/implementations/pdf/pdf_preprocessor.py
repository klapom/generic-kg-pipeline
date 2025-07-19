"""
PDF Preprocessor - Unified image extraction and caching
Handles all PDF preprocessing: page rendering and embedded image extraction
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import fitz  # PyMuPDF
from PIL import Image
import io
import logging

from core.parsers.interfaces.data_models import DocumentType

logger = logging.getLogger(__name__)


class PreprocessResult:
    """Result container for preprocessing operations"""
    def __init__(self):
        self.pdf_hash: str = ""
        self.page_images: Dict[int, str] = {}  # page_num -> image_path
        self.embedded_images: List[Dict[str, Any]] = []  # List of image metadata
        self.page_count: int = 0
        self.cache_dir: Optional[Path] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "pdf_hash": self.pdf_hash,
            "page_images": self.page_images,
            "embedded_images": self.embedded_images,
            "page_count": self.page_count,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None
        }
    
    @classmethod
    def from_cache(cls, cache_file: Path) -> 'PreprocessResult':
        """Load from cache file"""
        result = cls()
        with open(cache_file, 'r') as f:
            data = json.load(f)
            result.pdf_hash = data["pdf_hash"]
            result.page_images = data["page_images"]
            result.embedded_images = data["embedded_images"]
            result.page_count = data["page_count"]
            result.cache_dir = Path(data["cache_dir"]) if data["cache_dir"] else None
        return result


class PDFPreprocessor:
    """
    Preprocesses PDFs for efficient parsing:
    1. Renders pages as images for SmolDocling
    2. Extracts embedded images with position info
    3. Caches everything with deterministic filenames
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor
        
        Args:
            cache_dir: Directory for caching images
            config: Configuration options
        """
        self.config = config or {}
        self.cache_dir = cache_dir or Path("cache/images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.page_dpi = self.config.get("page_dpi", 150)  # DPI for page rendering
        self.max_pages = self.config.get("max_pages", 20)  # Max pages to process
        self.extract_embedded = self.config.get("extract_embedded", True)
        self.render_pages = self.config.get("render_pages", True)
        self.force_reprocess = self.config.get("force_reprocess", False)
        
    def get_pdf_hash(self, pdf_path: Path) -> str:
        """Generate hash for PDF file"""
        hasher = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]  # Use first 16 chars
    
    def get_image_path(self, pdf_hash: str, page: int, index: Optional[int] = None) -> Path:
        """
        Generate image path using naming convention
        
        Args:
            pdf_hash: Hash of the PDF
            page: Page number (0-based)
            index: Image index on page (None for full page)
            
        Returns:
            Path to image file
        """
        if index is None:
            # Full page image
            filename = f"{pdf_hash[:8]}_{page:03d}_full.png"
        else:
            # Embedded image
            filename = f"{pdf_hash[:8]}_{page:03d}_{index:03d}.png"
        
        return self.cache_dir / filename
    
    def preprocess(self, pdf_path: Path) -> PreprocessResult:
        """
        Main preprocessing method
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PreprocessResult with all extracted data
        """
        logger.info(f"🔧 Preprocessing PDF: {pdf_path}")
        
        # Calculate PDF hash
        pdf_hash = self.get_pdf_hash(pdf_path)
        logger.info(f"📊 PDF hash: {pdf_hash}")
        
        # Check cache
        cache_info_file = self.cache_dir / f"{pdf_hash[:8]}_info.json"
        if cache_info_file.exists() and not self.force_reprocess:
            logger.info("✅ Using cached preprocessing results")
            return PreprocessResult.from_cache(cache_info_file)
        
        # Process PDF
        result = PreprocessResult()
        result.pdf_hash = pdf_hash
        result.cache_dir = self.cache_dir
        
        try:
            doc = fitz.open(str(pdf_path))
            result.page_count = len(doc)
            
            pages_to_process = min(len(doc), self.max_pages)
            logger.info(f"📄 Processing {pages_to_process} of {len(doc)} pages")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                
                # 1. Render page as image (for SmolDocling)
                if self.render_pages:
                    page_image_path = self._render_page(
                        page, pdf_hash, page_num
                    )
                    result.page_images[page_num] = str(page_image_path)
                
                # 2. Extract embedded images
                if self.extract_embedded:
                    embedded = self._extract_page_images(
                        doc, page, pdf_hash, page_num
                    )
                    result.embedded_images.extend(embedded)
            
            doc.close()
            
            # Save cache info
            with open(cache_info_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
                
            logger.info(f"✅ Preprocessing complete: {len(result.page_images)} pages, "
                       f"{len(result.embedded_images)} embedded images")
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing PDF: {e}")
            raise
            
        return result
    
    def _render_page(self, page: fitz.Page, pdf_hash: str, page_num: int) -> Path:
        """Render a page as image"""
        image_path = self.get_image_path(pdf_hash, page_num)
        
        if image_path.exists() and not self.force_reprocess:
            logger.debug(f"✅ Page image exists: {image_path}")
            return image_path
        
        # Render page
        mat = fitz.Matrix(self.page_dpi / 72.0, self.page_dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Save as PNG
        pix.save(str(image_path))
        logger.debug(f"💾 Saved page image: {image_path}")
        
        return image_path
    
    def _extract_page_images(
        self, 
        doc: fitz.Document, 
        page: fitz.Page, 
        pdf_hash: str, 
        page_num: int
    ) -> List[Dict[str, Any]]:
        """Extract embedded images from a page"""
        embedded_images = []
        
        # Get image list
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract image
                xref = img[0]
                
                # Get image data
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image position on page
                bbox = self._get_image_bbox(page, xref)
                
                # Save image
                image_path = self.get_image_path(pdf_hash, page_num, img_index)
                
                if not image_path.exists() or self.force_reprocess:
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    logger.debug(f"💾 Saved embedded image: {image_path}")
                
                # Create metadata
                image_info = {
                    "path": str(image_path),
                    "page": page_num,
                    "index": img_index,
                    "bbox": bbox,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0),
                    "format": base_image.get("ext", "png"),
                    "hash_ref": f"{pdf_hash[:8]}_{page_num:03d}_{img_index:03d}"
                }
                
                embedded_images.append(image_info)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract image {img_index} on page {page_num}: {e}")
                
        return embedded_images
    
    def _get_image_bbox(self, page: fitz.Page, xref: int) -> Optional[Dict[str, float]]:
        """Get bounding box of image on page"""
        try:
            # Get all image instances on the page
            for item in page.get_image_bbox(xref):
                # Return first instance (usually there's only one)
                rect = item
                return {
                    "x": rect.x0,
                    "y": rect.y0,
                    "width": rect.width,
                    "height": rect.height
                }
        except:
            return None
    
    def cleanup_cache(self, keep_recent: int = 100):
        """Clean up old cached images"""
        # TODO: Implement cache cleanup based on access time
        pass