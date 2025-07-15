"""
Text file parser for plain text documents
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.parsers.interfaces import (
    BaseParser, Document, DocumentMetadata, DocumentType, 
    Segment, ParseError, SegmentType, TextSubtype
)

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = False):
        """Initialize TXT parser"""
        super().__init__(config)
        self.supported_types = {DocumentType.TXT}
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
                file_path=file_path,
                segments=segments,
                metadata=metadata,
                visual_elements=[]  # No visual elements in text files
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
        
        # Split by double newlines but keep the parts
        # We'll handle headers differently
        parts = re.split(r'\n\n+', content)
        
        position = 0
        segment_index = 0
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # For very short parts (like headers), allow them
            if len(part) < self.min_paragraph_length and not part.startswith('#'):
                continue
            
            # Determine segment type and subtype
            segment_type = SegmentType.TEXT
            segment_subtype = TextSubtype.PARAGRAPH
            
            if part.startswith('#'):
                # Count the number of # symbols
                header_match = re.match(r'^(#+)\s', part)
                if header_match:
                    level = len(header_match.group(1))
                    if level == 1:
                        segment_subtype = TextSubtype.HEADING_1
                    elif level == 2:
                        segment_subtype = TextSubtype.HEADING_2
                    elif level == 3:
                        segment_subtype = TextSubtype.HEADING_3
                    else:
                        # For levels 4-6, default to heading_3
                        segment_subtype = TextSubtype.HEADING_3
                    
                    logger.info(f"📝 Found header level {level}: {part[:50]}...")
            
            # Check for other patterns
            elif any(part.startswith(prefix) for prefix in ['- ', '* ', '+ ']) or re.match(r'^\d+\.', part):
                segment_subtype = TextSubtype.LIST
                logger.debug(f"📝 Found list: {part[:50]}...")
            elif part.startswith(('```', '~~~')):
                segment_subtype = TextSubtype.CODE
                logger.debug(f"📝 Found code block: {part[:50]}...")
            elif part.startswith('>'):
                segment_subtype = TextSubtype.QUOTE
                logger.debug(f"📝 Found quote: {part[:50]}...")
            
            # Create segment with new structure
            segment = Segment(
                content=part,
                page_number=1,  # All on "page 1" for text files
                segment_index=segment_index,
                segment_type=segment_type,
                segment_subtype=segment_subtype.value,
                metadata={
                    "index": i,
                    "length": len(part),
                    "position": (position, position + len(part)),
                    "source_format": "txt"
                }
            )
            
            segments.append(segment)
            segment_index += 1
            position += len(part) + 2  # Account for newlines
            
            logger.info(f"📄 Segment {segment_index + 1}: type={segment_type.value}, subtype={segment_subtype.value if hasattr(segment_subtype, 'value') else segment_subtype}, content_preview='{part[:30]}...', length={len(part)}")
        
        return segments