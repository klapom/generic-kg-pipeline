#!/usr/bin/env python3
"""Test visual elements extraction and VLM analysis"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime
import fitz  # PyMuPDF
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/visual_elements_with_vlm")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'test_run_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.insert(0, '.')

from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient
from core.parsers.vlm_integration import VLMIntegration, AdaptivePromptGenerator
from core.parsers.interfaces.data_models import Segment, SegmentType, VisualSubtype, VisualElement, VisualElementType, DocumentType
from core.config import get_config

def extract_visual_element_from_pdf(pdf_path: Path, page_num: int, bbox: List[int], output_path: Path):
    """Extract actual visual element from PDF using bbox coordinates"""
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Convert bbox from 0-500 scale to page coordinates
        scale_x = page_width / 500.0
        scale_y = page_height / 500.0
        
        x0 = bbox[0] * scale_x
        y0 = bbox[1] * scale_y
        x1 = bbox[2] * scale_x
        y1 = bbox[3] * scale_y
        
        # Create rectangle for the visual element
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # Create a pixmap of the area
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Save the image
        pix.save(output_path)
        
        doc.close()
        
        logger.info(f"Extracted visual element to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract visual element: {e}")
        return False

async def analyze_visual_elements():
    """Extract visual elements and analyze them with VLM"""
    
    # Initialize configuration
    config = get_config()
    
    # Initialize SmolDocling client
    client = VLLMSmolDoclingClient()
    
    # Initialize VLM integration
    vlm = VLMIntegration()
    prompt_generator = AdaptivePromptGenerator()
    
    # Parse PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    logger.info(f"Parsing PDF: {pdf_path}")
    
    # Parse the document
    result = client.parse_pdf(pdf_path)
    
    # Extract visual elements from all pages
    visual_elements = []
    image_dir = log_dir / "extracted_images"
    image_dir.mkdir(exist_ok=True)
    
    for page_num, page_data in enumerate(result.pages, 1):
        if hasattr(page_data, 'images') and page_data.images:
            for idx, image_elem in enumerate(page_data.images):
                if 'bbox' in image_elem and image_elem['bbox']:
                    # Extract the image
                    image_name = f"visual_page{page_num}_element{idx}.png"
                    image_path = image_dir / image_name
                    
                    if extract_visual_element_from_pdf(pdf_path, page_num, image_elem['bbox'], image_path):
                        visual_elements.append({
                            'page': page_num,
                            'element': image_elem,
                            'image_path': image_path,
                            'image_name': image_name
                        })
    
    logger.info(f"Extracted {len(visual_elements)} visual elements")
    
    # Analyze each visual element with VLM
    results = []
    for idx, ve in enumerate(visual_elements):
        logger.info(f"Analyzing {ve['image_name']} with VLM...")
        
        try:
            # Create Segment object for visual element
            visual_segment = Segment(
                content=f"[IMAGE: Page {ve['page']}, Element {idx}]",
                page_number=ve['page'],
                segment_index=idx,
                segment_type=SegmentType.VISUAL,
                segment_subtype=VisualSubtype.IMAGE.value,
                metadata={
                    'bbox': ve['element']['bbox'],
                    'confidence': ve['element'].get('confidence', 0.9),
                    'image_path': str(ve['image_path'])
                }
            )
            
            # Generate adaptive prompt
            prompt = prompt_generator.generate_prompt(
                element_type=VisualSubtype.IMAGE.value,
                document_context={
                    'page_number': ve['page'],
                    'element_type': 'image'
                }
            )
            
            # Create a proper VisualElement for VLM analysis
            # Load the image data as bytes
            with open(ve['image_path'], 'rb') as f:
                image_bytes = f.read()
                
            visual_element = VisualElement(
                element_type=VisualElementType.IMAGE,
                source_format=DocumentType.PDF,
                content_hash=VisualElement.create_hash(image_bytes),
                bounding_box={'bbox': ve['element']['bbox']},
                page_or_slide=ve['page'],
                raw_data=image_bytes
            )
            
            # Analyze with VLM (batch call with single element)
            analyzed_elements = await vlm.analyze_visual_elements(
                [visual_element],
                document_context={'page_number': ve['page'], 'image_path': str(ve['image_path'])}
            )
            description = analyzed_elements[0].vlm_description if analyzed_elements else "Error: No analysis returned"
            
            results.append({
                'page': ve['page'],
                'image_name': ve['image_name'],
                'image_path': str(ve['image_path']),
                'element_type': 'image',
                'bbox': ve['element']['bbox'],
                'vlm_description': description,
                'original_content': visual_segment.content
            })
            
            logger.info(f"VLM Description: {description[:100]}...")
            
        except Exception as e:
            logger.error(f"Failed to analyze {ve['image_name']}: {e}")
            results.append({
                'page': ve['page'],
                'image_name': ve['image_name'],
                'image_path': str(ve['image_path']),
                'element_type': 'image',
                'bbox': ve['element']['bbox'],
                'vlm_description': f"Error: {str(e)}",
                'original_content': f"[IMAGE: Page {ve['page']}]"
            })
    
    # Save results as JSON
    results_json = log_dir / "visual_analysis_results.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create HTML report
    create_html_report(results, log_dir)
    
    logger.info(f"Analysis complete. Results saved to {log_dir}")
    return results

def create_html_report(results: List[Dict[str, Any]], output_dir: Path):
    """Create an HTML report with images and VLM descriptions"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Visual Elements Analysis with VLM</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }
        h1 { 
            color: #333; 
            text-align: center;
        }
        .visual-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
            margin-top: 20px;
        }
        .visual-item { 
            background: white;
            border: 1px solid #ddd; 
            border-radius: 8px;
            padding: 15px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .visual-item h3 { 
            margin-top: 0; 
            color: #0066cc;
        }
        .visual-item img { 
            max-width: 100%; 
            height: auto; 
            border: 1px solid #eee;
            margin: 10px 0;
        }
        .metadata {
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 14px;
        }
        .vlm-description {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
            border-left: 3px solid #0066cc;
        }
        .original-content {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        .success { color: #28a745; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <h1>Visual Elements Analysis with VLM</h1>
    <p style="text-align: center;">Extracted {count} visual elements from BMW 3 Series G20 Preview document</p>
    <div class="visual-grid">
"""
    
    for result in results:
        status_class = "error" if result['vlm_description'].startswith("Error:") else "success"
        
        # Copy image to relative path for HTML
        image_relative = f"extracted_images/{result['image_name']}"
        
        html_content += f"""
        <div class="visual-item">
            <h3>{result['image_name']}</h3>
            <div class="metadata">
                <strong>Page:</strong> {result['page']}<br>
                <strong>Type:</strong> {result['element_type']}<br>
                <strong>BBox:</strong> {result['bbox']}
            </div>
            <img src="{image_relative}" alt="{result['image_name']}">
            <div class="original-content">
                <strong>Original Content:</strong> {result['original_content']}
            </div>
            <div class="vlm-description">
                <strong>VLM Analysis:</strong>
                <p class="{status_class}">{result['vlm_description']}</p>
            </div>
        </div>
        """
    
    html_content += """
    </div>
</body>
</html>
"""
    
    html_content = html_content.replace("{count}", str(len(results)))
    
    # Save HTML report
    html_path = output_dir / "visual_analysis_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report created: {html_path}")

if __name__ == "__main__":
    asyncio.run(analyze_visual_elements())