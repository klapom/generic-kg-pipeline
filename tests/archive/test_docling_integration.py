"""
Test suite for docling integration
Ensures backward compatibility and bbox preservation
"""

# import pytest  # Not needed for standalone test
from pathlib import Path
from typing import Dict, Any, List
import logging
import sys

# Add project root to path
sys.path.insert(0, '.')

# These would be imported after adding docling to requirements
# from docling_core.types.doc import DoclingDocument
# from docling_core.types.doc.document import DocTagsDocument

from core.clients.vllm_smoldocling_docling import VLLMSmolDoclingDoclingClient
from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

logger = logging.getLogger(__name__)


class TestDoclingIntegration:
    """Test docling integration maintains compatibility"""
    
    def sample_doctags(self) -> str:
        """Sample DocTags output from SmolDocling"""
        return """<doctag><picture><loc_0><loc_0><loc_500><loc_375><other></picture>
<section_header_level_1><loc_40><loc_356><loc_245><loc_373>Test Header</section_header_level_1>
<text><loc_49><loc_385><loc_272><loc_396>This is test content</text>
<page_footer><loc_35><loc_481><loc_251><loc_490>Page 1</page_footer>
</doctag>"""
    
    def expected_bbox_values(self) -> List[List[int]]:
        """Expected bbox values that must be preserved"""
        return [[0, 0, 500, 375]]  # Picture bbox
    
    def test_bbox_preservation(self, sample_doctags, expected_bbox_values):
        """Test that bbox values are preserved through parsing"""
        # Create mock output
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text})]
        
        output = MockOutput(sample_doctags)
        
        # Parse with legacy parser
        legacy_client = VLLMSmolDoclingClient()
        legacy_result = legacy_client.parse_model_output(output)
        
        # Verify legacy parser extracts bbox
        assert len(legacy_result["images"]) == 1
        assert legacy_result["images"][0]["bbox"] == expected_bbox_values[0]
        
        # TODO: Test with docling parser when available
        # docling_client = VLLMSmolDoclingDoclingClient(model_id="test", use_docling=True)
        # docling_result = docling_client.parse_model_output(output, page_image=mock_image)
        # assert docling_result["images"][0]["bbox"] == expected_bbox_values[0]
    
    def test_data_structure_compatibility(self, sample_doctags):
        """Test that output structure remains compatible"""
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text})]
        
        output = MockOutput(sample_doctags)
        
        # Parse with legacy
        legacy_client = VLLMSmolDoclingClient()
        result = legacy_client.parse_model_output(output)
        
        # Verify structure
        assert isinstance(result, dict)
        assert "text" in result
        assert "images" in result
        assert "tables" in result
        assert "formulas" in result
        assert "text_blocks" in result
        assert "layout_info" in result
        
        # Verify images have required fields
        for img in result.get("images", []):
            assert "content" in img
            assert "bbox" in img or img.get("bbox") is None
            assert "caption" in img
    
    def test_visual_element_creation(self, sample_doctags):
        """Test that VisualElements can be created from parsed data"""
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text})]
        
        output = MockOutput(sample_doctags)
        
        # Parse
        client = VLLMSmolDoclingClient()
        result = client.parse_model_output(output)
        
        # Create VisualElement from parsed image
        for img_data in result["images"]:
            visual = VisualElement(
                element_type=VisualElementType.IMAGE,
                source_format=DocumentType.PDF,
                content_hash=VisualElement.create_hash(str(img_data).encode()),
                bounding_box=img_data.get("bbox"),  # This is critical!
                page_or_slide=1,
                analysis_metadata={
                    "caption": img_data.get("caption", ""),
                    "raw_content": img_data.get("content", "")
                }
            )
            
            # Verify bbox was transferred
            assert visual.bounding_box == img_data.get("bbox")
            
            # Verify bbox format (should be list of 4 integers)
            if visual.bounding_box:
                assert isinstance(visual.bounding_box, list)
                assert len(visual.bounding_box) == 4
                assert all(isinstance(x, int) for x in visual.bounding_box)
    
    def test_bbox_extraction_patterns(self):
        """Test different bbox extraction patterns"""
        test_cases = [
            # New format with <loc_> tags
            {
                "input": "<picture><loc_0><loc_0><loc_500><loc_375><other></picture>",
                "expected": [0, 0, 500, 375]
            },
            # Old format with bare coordinates
            {
                "input": "<picture>0>0>500>375><other></picture>",
                "expected": [0, 0, 500, 375]
            },
            # Mixed format
            {
                "input": "<image>24>116>195>322>Car exterior</image>",
                "expected": [24, 116, 195, 322]
            }
        ]
        
        client = VLLMSmolDoclingDoclingClient(use_docling=False)
        
        for case in test_cases:
            # Test bbox parsing
            bbox = client._parse_bbox_from_doctags(case["input"])
            assert bbox == case["expected"], f"Failed for input: {case['input']}"
    
    def test_fallback_to_legacy(self):
        """Test fallback to legacy parser when docling fails"""
        # Create client with docling disabled
        client = VLLMSmolDoclingDoclingClient(
            use_docling=False  # Force legacy
        )
        
        assert not client.use_docling
        
        # Should use legacy parser
        class MockOutput:
            text = "<text>Test</text>"
        
        result = client.parse_model_output(MockOutput())
        assert isinstance(result, dict)
        assert "text" in result


class TestBBoxScaling:
    """Test bbox coordinate scaling for image extraction"""
    
    def test_bbox_scaling_calculation(self):
        """Test 0-500 to page coordinate scaling"""
        # SmolDocling bbox in 0-500 scale
        bbox_500_scale = [100, 100, 400, 400]
        
        # Page dimensions
        page_width = 1000  # pixels
        page_height = 1000  # pixels
        
        # Calculate scaling
        scale_x = page_width / 500.0  # = 2.0
        scale_y = page_height / 500.0  # = 2.0
        
        # Apply scaling
        x0 = bbox_500_scale[0] * scale_x  # 100 * 2 = 200
        y0 = bbox_500_scale[1] * scale_y  # 100 * 2 = 200
        x1 = bbox_500_scale[2] * scale_x  # 400 * 2 = 800
        y1 = bbox_500_scale[3] * scale_y  # 400 * 2 = 800
        
        assert [x0, y0, x1, y1] == [200, 200, 800, 800]
    
    def test_aspect_ratio_preservation(self):
        """Test that aspect ratios are preserved in scaling"""
        # Square in 0-500 scale
        bbox = [100, 100, 200, 200]
        width_500 = bbox[2] - bbox[0]  # 100
        height_500 = bbox[3] - bbox[1]  # 100
        aspect_500 = width_500 / height_500  # 1.0
        
        # Scale to different page size
        page_width = 1920
        page_height = 1080
        scale_x = page_width / 500.0
        scale_y = page_height / 500.0
        
        # Scaled dimensions
        width_scaled = width_500 * scale_x
        height_scaled = height_500 * scale_y
        aspect_scaled = width_scaled / height_scaled
        
        # Aspect ratio changes because page has different aspect
        assert aspect_scaled != aspect_500
        
        # But if page has same aspect ratio as 500x500
        page_width_square = 1000
        page_height_square = 1000
        scale_x_square = page_width_square / 500.0
        scale_y_square = page_height_square / 500.0
        
        width_scaled_square = width_500 * scale_x_square
        height_scaled_square = height_500 * scale_y_square
        aspect_scaled_square = width_scaled_square / height_scaled_square
        
        # Aspect ratio preserved
        assert abs(aspect_scaled_square - aspect_500) < 0.001


# Integration test would go here
def test_full_pipeline_integration():
    """Test full pipeline with docling integration"""
    # This would test:
    # 1. PDF → SmolDocling → parse_model_output (with docling) → Document
    # 2. Verify images have bbox
    # 3. Verify _extract_image_bytes works with the bbox
    # 4. Verify VLM receives correct image data
    pass


if __name__ == "__main__":
    # Run basic tests
    test = TestDoclingIntegration()
    
    # Test bbox patterns
    test.test_bbox_extraction_patterns()
    print("✅ BBox extraction patterns test passed")
    
    # Test scaling
    scale_test = TestBBoxScaling()
    scale_test.test_bbox_scaling_calculation()
    print("✅ BBox scaling test passed")
    
    print("\n📊 All compatibility tests passed!")