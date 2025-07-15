#!/usr/bin/env python3
"""
Comprehensive test for new segment structure across all parsers
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

from core.parsers.interfaces import (
    SegmentType, TextSubtype, VisualSubtype, TableSubtype, MetadataSubtype, 
    Document, Segment
)
from core.parsers.implementations.text import TXTParser
from core.parsers.implementations.pdf import HybridPDFParser
from core.parsers.implementations.office import DOCXParser, XLSXParser, PPTXParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentStructureValidator:
    """Validates new segment structure"""
    
    @staticmethod
    def validate_segment(segment: Segment) -> Dict[str, Any]:
        """Validate a single segment"""
        issues = []
        
        # Check segment_type is enum
        if not hasattr(segment.segment_type, 'value'):
            issues.append(f"segment_type is not an enum: {type(segment.segment_type)}")
        else:
            # Check valid segment type
            if segment.segment_type not in [SegmentType.TEXT, SegmentType.VISUAL, 
                                           SegmentType.TABLE, SegmentType.METADATA]:
                issues.append(f"Invalid segment_type: {segment.segment_type}")
        
        # Check segment_subtype exists
        if segment.segment_subtype is None:
            issues.append("segment_subtype is None")
        
        # Check content not empty
        if not segment.content or not segment.content.strip():
            issues.append("Empty content")
        
        # Check segment_index
        if not hasattr(segment, 'segment_index'):
            issues.append("Missing segment_index")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "segment_type": segment.segment_type.value if hasattr(segment.segment_type, 'value') else str(segment.segment_type),
            "segment_subtype": segment.segment_subtype,
            "content_preview": segment.content[:50] + "..." if len(segment.content) > 50 else segment.content
        }
    
    @staticmethod
    def validate_document(document: Document) -> Dict[str, Any]:
        """Validate entire document"""
        total_segments = len(document.segments)
        valid_segments = 0
        invalid_segments = []
        
        segment_type_counts = {}
        subtype_counts = {}
        
        for i, segment in enumerate(document.segments):
            validation = SegmentStructureValidator.validate_segment(segment)
            
            if validation["valid"]:
                valid_segments += 1
                
                # Count types
                seg_type = validation["segment_type"]
                seg_subtype = validation["segment_subtype"]
                
                segment_type_counts[seg_type] = segment_type_counts.get(seg_type, 0) + 1
                subtype_counts[f"{seg_type}/{seg_subtype}"] = subtype_counts.get(f"{seg_type}/{seg_subtype}", 0) + 1
            else:
                invalid_segments.append({
                    "index": i,
                    "issues": validation["issues"],
                    "preview": validation["content_preview"]
                })
        
        # Check visual elements
        visual_segment_count = sum(1 for seg in document.segments if seg.segment_type == SegmentType.VISUAL)
        visual_element_count = len(document.visual_elements)
        
        return {
            "total_segments": total_segments,
            "valid_segments": valid_segments,
            "invalid_segments": invalid_segments,
            "segment_type_counts": segment_type_counts,
            "subtype_counts": subtype_counts,
            "visual_segments": visual_segment_count,
            "visual_elements": visual_element_count,
            "visual_consistency": visual_segment_count == visual_element_count
        }


async def test_txt_parser():
    """Test TXT parser with new structure"""
    logger.info("\n=== Testing TXT Parser ===")
    
    # Create test file
    test_file = Path("/tmp/test_txt_structure.txt")
    test_content = """# Main Title

This is a paragraph.

## Section

- List item 1
- List item 2

> Quote block

```python
code block
```
"""
    
    test_file.write_text(test_content)
    
    try:
        parser = TXTParser()
        document = await parser.parse(test_file)
        
        validation = SegmentStructureValidator.validate_document(document)
        logger.info(f"TXT Parser Results: {json.dumps(validation, indent=2)}")
        
        return validation
        
    finally:
        test_file.unlink()


async def test_docx_parser():
    """Test DOCX parser with new structure"""
    logger.info("\n=== Testing DOCX Parser ===")
    
    test_file = Path("data/input/Erkenntnisse_Herausforderungen.docx")
    if not test_file.exists():
        logger.warning("No DOCX test file found")
        return None
    
    parser = DOCXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    validation = SegmentStructureValidator.validate_document(document)
    logger.info(f"DOCX Parser Results: {json.dumps(validation, indent=2)}")
    
    return validation


async def test_xlsx_parser():
    """Test XLSX parser with new structure"""
    logger.info("\n=== Testing XLSX Parser ===")
    
    test_file = Path("data/input/PapersWithCode_Abstracts.xlsx")
    if not test_file.exists():
        logger.warning("No XLSX test file found")
        return None
    
    parser = XLSXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    validation = SegmentStructureValidator.validate_document(document)
    logger.info(f"XLSX Parser Results: {json.dumps(validation, indent=2)}")
    
    return validation


async def test_pptx_parser():
    """Test PPTX parser with new structure"""
    logger.info("\n=== Testing PPTX Parser ===")
    
    test_file = Path("data/input/CRISP-DM_TätigkeitenTools.pptx")
    if not test_file.exists():
        logger.warning("No PPTX test file found")
        return None
    
    parser = PPTXParser(enable_vlm=False)
    document = await parser.parse(test_file)
    
    validation = SegmentStructureValidator.validate_document(document)
    logger.info(f"PPTX Parser Results: {json.dumps(validation, indent=2)}")
    
    return validation


async def test_pdf_parser():
    """Test PDF parser with new structure"""
    logger.info("\n=== Testing PDF Parser ===")
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not test_file.exists():
        logger.warning("No PDF test file found")
        return None
    
    # Test with mode 0 (SmolDocling only, no visual segments for now)
    parser = HybridPDFParser(config={"pdfplumber_mode": 0}, enable_vlm=False)
    document = await parser.parse(test_file)
    
    validation = SegmentStructureValidator.validate_document(document)
    logger.info(f"PDF Parser Results: {json.dumps(validation, indent=2)}")
    
    return validation


async def run_all_tests():
    """Run tests for all parsers"""
    logger.info("🚀 Starting comprehensive segment structure tests")
    
    results = {
        "txt": await test_txt_parser(),
        "docx": await test_docx_parser(),
        "xlsx": await test_xlsx_parser(),
        "pptx": await test_pptx_parser(),
        # Skip PDF test due to long vLLM initialization time
        # "pdf": await test_pdf_parser()
    }
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    all_valid = True
    
    for parser_name, result in results.items():
        if result:
            valid = result["valid_segments"] == result["total_segments"]
            all_valid = all_valid and valid
            
            logger.info(f"\n{parser_name.upper()} Parser:")
            logger.info(f"  ✅ Valid segments: {result['valid_segments']}/{result['total_segments']}")
            
            if result["invalid_segments"]:
                logger.error(f"  ❌ Invalid segments: {len(result['invalid_segments'])}")
                for invalid in result["invalid_segments"][:3]:  # Show first 3
                    logger.error(f"    - Segment {invalid['index']}: {invalid['issues']}")
            
            logger.info(f"  📊 Type distribution: {result['segment_type_counts']}")
            logger.info(f"  📊 Visual consistency: {result['visual_consistency']}")
    
    if all_valid:
        logger.info("\n✅ ALL TESTS PASSED! New segment structure is working correctly.")
    else:
        logger.error("\n❌ SOME TESTS FAILED! Check the errors above.")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())