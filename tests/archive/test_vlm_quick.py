#!/usr/bin/env python3
"""
Quick test for VLM integration components without full model loading
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.vlm_integration import AdaptivePromptGenerator, VLMIntegration
from core.parsers.interfaces import (
    VisualElement, VisualElementType, Segment, 
    SegmentType, VisualSubtype, DocumentType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_adaptive_prompt_generator():
    """Test the adaptive prompt generator"""
    logger.info("🧪 Testing AdaptivePromptGenerator")
    
    generator = AdaptivePromptGenerator()
    
    # Test different element types
    element_types = [
        VisualElementType.CHART,
        VisualElementType.DIAGRAM,
        VisualElementType.FORMULA,
        VisualElementType.TABLE,
        VisualElementType.SCREENSHOT,
        VisualElementType.IMAGE
    ]
    
    document_context = {
        "document_title": "BMW 3 Series Technical Preview",
        "section": "Performance Metrics",
        "document_type": "Technical Documentation"
    }
    
    for elem_type in element_types:
        logger.info(f"\n  Testing prompt for: {elem_type.value}")
        
        prompt_config = generator.generate_prompt(elem_type, document_context)
        
        logger.info(f"    Analysis focus: {prompt_config['analysis_focus']}")
        logger.info(f"    Instructions: {len(prompt_config['context'].get('instructions', []))} items")
        
        # Show first instruction
        instructions = prompt_config['context'].get('instructions', [])
        if instructions:
            logger.info(f"    First instruction: {instructions[0]}")
    
    logger.info("\n✅ AdaptivePromptGenerator test passed")


def test_visual_segment_creation():
    """Test creating visual segments with placeholders"""
    logger.info("\n🧪 Testing Visual Segment Creation")
    
    # Create test visual elements
    visual_elements = [
        VisualElement(
            element_type=VisualElementType.CHART,
            source_format=DocumentType.PDF,
            content_hash="chart_hash_123",
            confidence=0.95,
            bounding_box={"x": 100, "y": 200, "width": 200, "height": 200},
            page_or_slide=1
        ),
        VisualElement(
            element_type=VisualElementType.DIAGRAM,
            source_format=DocumentType.PDF,
            content_hash="diagram_hash_456",
            confidence=0.90,
            bounding_box={"x": 150, "y": 250, "width": 200, "height": 200},
            page_or_slide=2
        ),
        VisualElement(
            element_type=VisualElementType.FORMULA,
            source_format=DocumentType.PDF,
            content_hash="formula_hash_789",
            confidence=0.85,
            bounding_box={"x": 200, "y": 300, "width": 200, "height": 50},
            page_or_slide=3
        )
    ]
    
    # Create visual segments
    visual_segments = []
    for i, ve in enumerate(visual_elements):
        # Map element type to visual subtype
        subtype_map = {
            VisualElementType.CHART: VisualSubtype.CHART,
            VisualElementType.DIAGRAM: VisualSubtype.DIAGRAM,
            VisualElementType.FORMULA: VisualSubtype.FORMULA,
            VisualElementType.TABLE_IMAGE: VisualSubtype.IMAGE,  # Table images are treated as images
            VisualElementType.IMAGE: VisualSubtype.IMAGE,
            VisualElementType.SCREENSHOT: VisualSubtype.SCREENSHOT
        }
        
        subtype = subtype_map.get(ve.element_type, VisualSubtype.IMAGE).value
        content = f"[{ve.element_type.value.upper()}: Placeholder for visual analysis]"
        
        segment = Segment(
            content=content,
            segment_type=SegmentType.VISUAL,
            segment_subtype=subtype,
            page_number=ve.page_or_slide,
            segment_index=i,
            visual_references=[ve.content_hash],
            metadata={
                "placeholder": True,
                "bounding_box": ve.bounding_box
            }
        )
        
        visual_segments.append(segment)
        
        logger.info(f"\n  Created visual segment {i+1}:")
        logger.info(f"    Type: {segment.segment_type.value}/{segment.segment_subtype}")
        logger.info(f"    Content: {segment.content}")
        logger.info(f"    Page: {segment.page_number}")
        logger.info(f"    Visual ref: {segment.visual_references[0]}")
    
    logger.info(f"\n✅ Created {len(visual_segments)} visual segments with placeholders")
    return visual_segments, visual_elements


def test_segment_update_simulation():
    """Simulate updating visual segments with VLM descriptions"""
    logger.info("\n🧪 Testing Visual Segment Update Simulation")
    
    # Create test segments and elements
    visual_segments, visual_elements = test_visual_segment_creation()
    
    # Simulate VLM descriptions
    vlm_descriptions = {
        "chart_hash_123": "Bar chart showing BMW 3 Series sales data from 2020-2023. The chart displays quarterly sales figures with blue bars representing sedan models and red bars for touring variants. Peak sales of 45,000 units occurred in Q3 2022.",
        "diagram_hash_456": "Technical diagram illustrating the BMW TwinPower Turbo engine layout. Shows the twin-scroll turbocharger positioning, direct fuel injection system, and variable valve timing components. Key components are labeled with part numbers.",
        "formula_hash_789": "Mathematical formula: P = τ × ω, where P represents power output in watts, τ is torque in Newton-meters, and ω is angular velocity in radians per second."
    }
    
    # Update visual elements with VLM descriptions
    for ve in visual_elements:
        if ve.content_hash in vlm_descriptions:
            ve.vlm_description = vlm_descriptions[ve.content_hash]
            ve.confidence = 0.95
    
    # Update segments
    visual_map = {ve.content_hash: ve for ve in visual_elements}
    
    logger.info("\n  Updating segments with VLM descriptions:")
    
    for segment in visual_segments:
        if segment.visual_references:
            hash_ref = segment.visual_references[0]
            
            if hash_ref in visual_map:
                visual_elem = visual_map[hash_ref]
                
                if visual_elem.vlm_description:
                    # Update content with VLM description
                    old_content = segment.content
                    segment.content = visual_elem.vlm_description
                    
                    # Update metadata
                    segment.metadata.update({
                        "vlm_analyzed": True,
                        "confidence": visual_elem.confidence,
                        "original_placeholder": old_content
                    })
                    
                    logger.info(f"\n  Updated segment on page {segment.page_number}:")
                    logger.info(f"    Old: {old_content}")
                    logger.info(f"    New: {segment.content[:80]}...")
                    logger.info(f"    Confidence: {segment.metadata['confidence']}")
    
    logger.info("\n✅ Visual segment update simulation completed")
    
    # Save example output
    output_dir = Path("tests/debugging/vlm_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    segments_data = []
    for seg in visual_segments:
        seg_data = {
            "type": seg.segment_type.value,
            "subtype": seg.segment_subtype,
            "content": seg.content,
            "page": seg.page_number,
            "visual_ref": seg.visual_references[0] if seg.visual_references else None,
            "vlm_analyzed": seg.metadata.get("vlm_analyzed", False)
        }
        segments_data.append(seg_data)
    
    example_file = output_dir / "vlm_example_segments.json"
    with open(example_file, 'w', encoding='utf-8') as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 Example saved to: {example_file}")


def main():
    """Run quick VLM integration tests"""
    logger.info("=" * 80)
    logger.info("VLM INTEGRATION QUICK TEST")
    logger.info("=" * 80)
    
    # Test 1: Adaptive prompt generator
    test_adaptive_prompt_generator()
    
    # Test 2: Visual segment update simulation
    test_segment_update_simulation()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ VLM INTEGRATION QUICK TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()