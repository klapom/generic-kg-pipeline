#!/usr/bin/env python3
"""Demo test for VLM analysis with just a few images"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
log_dir = Path("tests/debugging/vlm_demo")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'demo_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.insert(0, '.')

from core.parsers.vlm_integration import VLMIntegration
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

async def demo_vlm_analysis():
    """Demo VLM analysis with a few pre-extracted images"""
    
    # Initialize VLM integration
    logger.info("Initializing VLM integration...")
    vlm = VLMIntegration()
    
    # Select just a few interesting images
    test_images = [
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
            "description": "BMW front view with annotations"
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
            "description": "BMW interior dashboard"
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element1.png"),
            "description": "BMW interior seats"
        }
    ]
    
    results = []
    
    for idx, img_info in enumerate(test_images):
        if not img_info["path"].exists():
            logger.warning(f"Image not found: {img_info['path']}")
            continue
            
        logger.info(f"Analyzing image {idx + 1}/{len(test_images)}: {img_info['description']}")
        
        try:
            # Load image bytes
            with open(img_info["path"], 'rb') as f:
                image_bytes = f.read()
            
            # Create VisualElement
            visual_element = VisualElement(
                element_type=VisualElementType.IMAGE,
                source_format=DocumentType.PDF,
                content_hash=VisualElement.create_hash(image_bytes),
                page_or_slide=idx + 1,
                raw_data=image_bytes
            )
            
            # Analyze with VLM
            analyzed_elements = await vlm.analyze_visual_elements(
                [visual_element],
                document_context={
                    'document_type': 'BMW 3 Series brochure',
                    'description': img_info['description']
                }
            )
            
            if analyzed_elements:
                result = {
                    'image': str(img_info['path'].name),
                    'description': img_info['description'],
                    'vlm_analysis': analyzed_elements[0].vlm_description,
                    'confidence': analyzed_elements[0].confidence,
                    'extracted_data': analyzed_elements[0].extracted_data
                }
                results.append(result)
                
                logger.info(f"VLM Analysis: {result['vlm_analysis'][:200]}...")
                logger.info(f"Confidence: {result['confidence']:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to analyze {img_info['path']}: {e}")
            results.append({
                'image': str(img_info['path'].name),
                'description': img_info['description'],
                'error': str(e)
            })
    
    # Save results
    results_file = log_dir / "vlm_demo_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Create simple HTML report
    create_demo_report(results, log_dir)
    
    logger.info(f"Demo completed. Results saved to {log_dir}")
    return results

def create_demo_report(results, output_dir):
    """Create a simple HTML report for the demo"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>VLM Demo Results</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            max-width: 1200px;
            margin: 0 auto;
        }
        .result-item {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #f9f9f9;
        }
        .image-name {
            font-weight: bold;
            color: #0066cc;
            font-size: 18px;
        }
        .description {
            color: #666;
            margin: 10px 0;
        }
        .vlm-analysis {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 3px solid #0066cc;
        }
        .confidence {
            color: #28a745;
            font-weight: bold;
        }
        .error {
            color: #dc3545;
            background: #ffe8e8;
            padding: 10px;
            border-radius: 4px;
        }
        .extracted-data {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>VLM Demo Results - BMW 3 Series Analysis</h1>
    <p>Analyzed {count} images using Qwen2.5-VL-7B</p>
"""
    
    for result in results:
        if 'error' in result:
            html_content += f"""
    <div class="result-item">
        <div class="image-name">{result['image']}</div>
        <div class="description">{result['description']}</div>
        <div class="error">Error: {result['error']}</div>
    </div>
"""
        else:
            extracted_data_str = json.dumps(result.get('extracted_data', {}), indent=2) if result.get('extracted_data') else "None"
            
            html_content += f"""
    <div class="result-item">
        <div class="image-name">{result['image']}</div>
        <div class="description">{result['description']}</div>
        <div class="vlm-analysis">
            <strong>VLM Analysis:</strong><br>
            {result.get('vlm_analysis', 'No analysis available')}
        </div>
        <div class="confidence">Confidence: {result.get('confidence', 0):.2%}</div>
        {f'<div class="extracted-data"><strong>Extracted Data:</strong><br>{extracted_data_str}</div>' if extracted_data_str != "None" else ''}
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_content = html_content.replace("{count}", str(len(results)))
    
    # Save HTML report
    html_path = output_dir / "vlm_demo_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report created: {html_path}")

if __name__ == "__main__":
    asyncio.run(demo_vlm_analysis())