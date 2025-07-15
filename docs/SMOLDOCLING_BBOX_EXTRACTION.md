# SmolDocling Bounding Box Extraction Guide

## Overview

SmolDocling outputs visual elements (pictures, diagrams, charts) with bounding box coordinates in a 0-500 scale format:

```
<picture><loc_424><loc_122><loc_490><loc_166></picture>
```

These coordinates represent: `<loc_x1><loc_y1><loc_x2><loc_y2>` where:
- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner
- Scale: 0-500 for both width and height

## Implementation Details

### 1. Parsing Picture Tags

The `parse_model_output` method in `vllm_smoldocling_local.py` extracts these coordinates:

```python
# Extract location coordinates from picture tags
loc_match = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', pic)
if loc_match:
    coords = tuple(map(int, loc_match.groups()))
    pictures.append({
        "content": pic.strip(),
        "caption": caption_match.group(1).strip() if caption_match else "",
        "bbox": list(coords)  # Store as [x1, y1, x2, y2]
    })
```

### 2. Storing in VisualElement Objects

The bounding boxes are stored in `VisualElement` objects with proper conversion:

```python
visual_elements.append(VisualElement(
    element_type=VisualElementType.FIGURE,
    source_format=DocumentType.PDF,
    content_hash=VisualElement.create_hash(...),
    vlm_description=image.get("description", ""),
    bounding_box={
        "x": bbox[0], 
        "y": bbox[1], 
        "width": bbox[2] - bbox[0], 
        "height": bbox[3] - bbox[1]
    },
    page_or_slide=page_number,
    analysis_metadata={
        "caption": image.get("caption"),
        "extracted_by": "SmolDocling",
        "raw_bbox": image.get("bbox")  # Original 0-500 scale coordinates
    }
))
```

### 3. Scale Conversion

A utility method `convert_bbox_scale` is provided to convert from the 0-500 scale to actual pixel coordinates:

```python
def convert_bbox_scale(bbox: List[int], target_width: int, target_height: int) -> Dict[str, float]:
    """Convert bbox from 0-500 scale to pixel coordinates"""
    scale_x = target_width / 500.0
    scale_y = target_height / 500.0
    
    x1, y1, x2, y2 = bbox
    
    return {
        "x": x1 * scale_x,
        "y": y1 * scale_y,
        "width": (x2 - x1) * scale_x,
        "height": (y2 - y1) * scale_y
    }
```

## Usage Examples

### Accessing Bounding Boxes

```python
# After parsing a PDF
document = client.convert_to_document(result, pdf_path)

# Iterate through visual elements
for visual_elem in document.visual_elements:
    if visual_elem.bounding_box:
        print(f"Element on page {visual_elem.page_or_slide}")
        print(f"Position: x={visual_elem.bounding_box['x']}, y={visual_elem.bounding_box['y']}")
        print(f"Size: {visual_elem.bounding_box['width']}x{visual_elem.bounding_box['height']}")
        
        # Access raw 0-500 scale coordinates if needed
        if visual_elem.analysis_metadata.get("raw_bbox"):
            raw = visual_elem.analysis_metadata["raw_bbox"]
            print(f"Raw coords (0-500 scale): {raw}")
```

### Converting to Different Scales

```python
# Get page dimensions (example: A4 at 300 DPI)
page_width_pixels = 2480  # 8.27 inches * 300 DPI
page_height_pixels = 3508  # 11.69 inches * 300 DPI

# Convert bbox to actual page pixels
if visual_elem.analysis_metadata.get("raw_bbox"):
    raw_bbox = visual_elem.analysis_metadata["raw_bbox"]
    pixel_bbox = client.convert_bbox_scale(raw_bbox, page_width_pixels, page_height_pixels)
    print(f"Pixel coordinates: {pixel_bbox}")
```

## Debugging

The system includes logging for bbox extraction:

```
DEBUG - Found 3/3 pictures with bounding boxes
DEBUG -   Picture 1: bbox=[424, 122, 490, 166] (0-500 scale)
DEBUG -   Picture 2: bbox=[100, 200, 300, 400] (0-500 scale)
DEBUG -   Picture 3: bbox=[50, 50, 150, 150] (0-500 scale)
```

## Notes

1. **Coordinate System**: SmolDocling uses a normalized 0-500 coordinate system regardless of actual page size
2. **Storage Format**: The `bounding_box` field stores x, y, width, height while `raw_bbox` in metadata stores the original [x1, y1, x2, y2]
3. **Deduplication**: The parser automatically deduplicates pictures with identical bounding boxes
4. **Missing Boxes**: Not all visual elements may have bounding boxes - always check if `bounding_box` is not None