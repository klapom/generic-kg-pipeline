"""
Text file parser for plain text documents
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_parser import BaseParser, Document, DocumentMetadata, DocumentType, Segment, ParseError

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = False):
        """Initialize TXT parser"""
        super().__init__(config)
        self.supported_extensions = ['.txt', '.text', '.md', '.markdown']
        self.enable_vlm = enable_vlm  # Not used for text files, but needed for compatibility
        
        # Text processing configuration
        self.min_paragraph_length = self.config.get("min_paragraph_length", 20)
        self.max_segment_length = self.config.get("max_segment_length", 1000)
        
        logger.info("Initialized TXT parser")
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a text file"""
        return file_path.suffix.lower() in self.supported_extensions
    
    async def parse(self, file_path: Path) -> Document:
        """
        Parse text document
        
        Args:
            file_path: Path to text file
            
        Returns:
            Document with text segments
        """
        self.validate_file(file_path)
        
        try:
            logger.info(f"🔍 Starting TXT parsing: {file_path.name}")
            
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, content)
            
            # Split into segments
            segments = self._create_segments(content)
            
            logger.info(f"✅ Created {len(segments)} segments from text file")
            
            # Create document
            document = self.create_document(
                content=content,
                segments=segments,
                visual_elements=[],  # No visual elements in text files
                metadata=metadata
            )
            
            logger.info(f"✅ TXT parsing completed: {file_path.name}")
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to parse TXT file {file_path}: {e}")
            raise ParseError(f"TXT parsing failed: {str(e)}")
    
    def _extract_metadata(self, file_path: Path, content: str) -> DocumentMetadata:
        """Extract metadata from text file"""
        # Count "pages" as roughly 3000 characters per page
        page_count = max(1, len(content) // 3000)
        
        # Try to extract title from first line or filename
        lines = content.strip().split('\n')
        title = file_path.stem
        
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a title (markdown header or short line)
            if first_line.startswith('#') or (len(first_line) < 100 and first_line):
                title = first_line.lstrip('#').strip()
        
        return DocumentMetadata(
            title=title,
            author="Unknown",
            created_date=datetime.now(),
            page_count=page_count,
            file_size=file_path.stat().st_size,
            file_path=file_path,
            document_type=DocumentType.TXT
        )
    
    def _create_segments(self, content: str) -> List[Segment]:
        """Create text segments from content"""
        segments = []
        
        # Split by double newlines (paragraphs) or headers
        parts = re.split(r'\n\n+|(?=^#{1,6}\s)', content, flags=re.MULTILINE)
        
        position = 0
        for i, part in enumerate(parts):
            part = part.strip()
            if not part or len(part) < self.min_paragraph_length:
                continue
            
            # Determine segment type
            segment_type = "paragraph"
            if part.startswith('#'):
                level = len(part.split()[0])
                segment_type = f"header_{level}"
                logger.info(f"📝 Found header level {level}: {part[:50]}...")
            
            # Create segment
            segment = Segment(
                content=part,
                page_number=1,  # All on "page 1" for text files
                segment_index=i,
                segment_type=segment_type,
                metadata={
                    "index": i,
                    "length": len(part),
                    "position": (position, position + len(part))
                }
            )
            
            segments.append(segment)
            position += len(part) + 2  # Account for newlines
            
            logger.info(f"📄 Segment {i+1}: {segment_type}, {len(part)} chars")
        
        return segments