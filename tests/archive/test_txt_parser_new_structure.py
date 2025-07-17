#!/usr/bin/env python3
"""Test TXT Parser with new segment structure"""

import asyncio
import logging
from pathlib import Path
import json

from core.parsers.implementations.text import TXTParser
from core.parsers.interfaces import SegmentType, TextSubtype

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_txt_parser():
    """Test TXT parser with new segment structure"""
    
    # Create test file
    test_file = Path("/tmp/test_segment_structure.txt")
    test_content = """# Main Title

This is a paragraph with some text content.

## Section Header

Another paragraph here.

- List item 1
- List item 2
- List item 3

### Subsection

> This is a quote block
> with multiple lines

```python
def hello():
    print("Hello World")
```

Final paragraph.
"""
    
    # Write test file
    test_file.write_text(test_content)
    logger.info(f"Created test file: {test_file}")
    
    try:
        # Initialize parser
        parser = TXTParser()
        
        # Parse file
        document = await parser.parse(test_file)
        
        logger.info(f"✅ Parsed document: {document.document_id}")
        logger.info(f"Total segments: {len(document.segments)}")
        
        # Display segments with new structure
        for i, segment in enumerate(document.segments):
            logger.info(f"\n--- Segment {i+1} ---")
            logger.info(f"Type: {segment.segment_type.value if hasattr(segment.segment_type, 'value') else segment.segment_type}")
            logger.info(f"Subtype: {segment.segment_subtype}")
            logger.info(f"Legacy type: {getattr(segment, '_legacy_segment_type', 'None')}")
            logger.info(f"Content preview: {segment.content[:50]}...")
            logger.info(f"Metadata: {json.dumps(segment.metadata, indent=2)}")
        
        # Verify segment types
        expected_subtypes = [
            TextSubtype.HEADING_1.value,    # # Main Title
            TextSubtype.PARAGRAPH.value,     # This is a paragraph...
            TextSubtype.HEADING_2.value,     # ## Section Header
            TextSubtype.PARAGRAPH.value,     # Another paragraph...
            TextSubtype.LIST.value,          # - List items...
            TextSubtype.HEADING_3.value,     # ### Subsection
            TextSubtype.QUOTE.value,         # > This is a quote...
            TextSubtype.CODE.value,          # ```python...
            TextSubtype.PARAGRAPH.value,     # Final paragraph.
        ]
        
        actual_subtypes = [seg.segment_subtype for seg in document.segments]
        
        logger.info(f"\n🔍 Verification:")
        logger.info(f"Expected subtypes: {expected_subtypes}")
        logger.info(f"Actual subtypes: {actual_subtypes}")
        
        # Check if all segments have the new structure
        all_have_new_structure = all(
            hasattr(seg, 'segment_type') and 
            hasattr(seg, 'segment_subtype') and
            seg.segment_type == SegmentType.TEXT
            for seg in document.segments
        )
        
        if all_have_new_structure:
            logger.info("✅ All segments have the new structure!")
        else:
            logger.error("❌ Some segments are missing the new structure!")
        
        return document
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
            logger.info("Cleaned up test file")


if __name__ == "__main__":
    asyncio.run(test_txt_parser())