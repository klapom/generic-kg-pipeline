#!/usr/bin/env python3
"""
Test PDF image extraction with bounding boxes
"""

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from pathlib import Path
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pdf_extraction():
    """Test extracting images from PDF using bbox coordinates"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    output_dir = Path("tests/debugging/pdf_extraction_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test bbox coordinates from SmolDocling (0-500 scale)
    test_bboxes = [
        {"page": 3, "bbox": [24, 116, 195, 322], "name": "exterior_1"},
        {"page": 3, "bbox": [290, 116, 493, 322], "name": "exterior_2"},
        {"page": 5, "bbox": [25, 115, 225, 476], "name": "interior_1"},
        {"page": 5, "bbox": [275, 115, 493, 476], "name": "interior_2"},
    ]
    
    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    for test in test_bboxes:
        page_num = test["page"] - 1  # Convert to 0-indexed
        bbox = test["bbox"]
        name = test["name"]
        
        logger.info(f"\nExtracting {name} from page {test['page']}:")
        logger.info(f"  BBox (0-500): {bbox}")
        
        # Get page
        page = doc[page_num]
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        logger.info(f"  Page dimensions: {page_width:.0f}x{page_height:.0f}")
        
        # Convert bbox from 0-500 scale to page coordinates
        scale_x = page_width / 500.0
        scale_y = page_height / 500.0
        
        x1, y1, x2, y2 = bbox
        rect = fitz.Rect(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        )
        
        logger.info(f"  Scaled rect: {rect}")
        
        # Extract the region
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        logger.info(f"  Extracted image size: {img.size}")
        
        # Add metadata overlay
        overlay_height = 60
        new_img = Image.new('RGB', (img.width, img.height + overlay_height), color='white')
        new_img.paste(img, (0, overlay_height))
        
        draw = ImageDraw.Draw(new_img)
        draw.rectangle([0, 0, img.width, overlay_height], fill='#e3f2fd')
        draw.text((10, 10), f"Page {test['page']} - {name}", fill='black')
        draw.text((10, 30), f"BBox: {bbox} (0-500 scale)", fill='black')
        
        # Save
        output_path = output_dir / f"{name}_page{test['page']}.png"
        new_img.save(output_path)
        logger.info(f"  ✅ Saved to: {output_path}")
    
    doc.close()
    
    # Create comparison HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Image Extraction Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .image-item { border: 1px solid #ccc; padding: 10px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>PDF Image Extraction Test Results</h1>
        <p>Successfully extracted images using SmolDocling bbox coordinates!</p>
        <div class="image-grid">
    """
    
    for test in test_bboxes:
        html_content += f"""
        <div class="image-item">
            <h3>{test['name'].replace('_', ' ').title()}</h3>
            <p>Page {test['page']}, BBox: {test['bbox']}</p>
            <img src="{test['name']}_page{test['page']}.png">
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    html_path = output_dir / "extraction_test.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"\n✅ Test complete! View results at: {html_path}")


if __name__ == "__main__":
    test_pdf_extraction()