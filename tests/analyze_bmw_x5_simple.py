#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
import json
import base64
import time
import logging
from datetime import datetime
import pymupdf  # PyMuPDF for direct PDF processing
from PIL import Image
import io
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_images_from_pdf(pdf_path):
    """Extract images directly from PDF using PyMuPDF"""
    images = []
    
    doc = pymupdf.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract images
        image_list = page.get_images(full=True)
        
        if image_list:
            logger.info(f"Page {page_num + 1}: Found {len(image_list)} images")
            
            for img_index, img in enumerate(image_list):
                # Get the XREF of the image
                xref = img[0]
                
                # Extract image data
                pix = pymupdf.Pixmap(doc, xref)
                
                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                images.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'data': img_data,
                    'type': 'extracted_image'
                })
                
                pix = None
        
        # If no images found, render the page as an image
        if not image_list or page_num == len(doc) - 1:  # Always render last page
            logger.info(f"Page {page_num + 1}: Rendering full page as image")
            mat = pymupdf.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            images.append({
                'page': page_num + 1,
                'index': 0,
                'data': img_data,
                'type': 'full_page'
            })
            
            pix = None
    
    doc.close()
    return images

def analyze_images_with_vlms(images):
    """Analyze extracted images with all VLMs"""
    
    output_dir = Path("tests/debugging/bmw_x5_vlm_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize VLM clients
    clients = {
        "Qwen2.5-VL-7B": TransformersQwen25VLClient(temperature=0.2, max_new_tokens=512),
        "LLaVA-1.6-Mistral-7B": TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.2,
            max_new_tokens=512
        ),
        "Pixtral-12B": TransformersPixtralClient(
            temperature=0.3,
            max_new_tokens=512,
            load_in_8bit=True
        )
    }
    
    results = []
    
    # Focus on interesting pages
    interesting_pages = [17]  # Just the diagram page for now
    
    for img_info in images:
        if img_info['page'] not in interesting_pages:
            continue
            
        logger.info(f"\n{'='*80}")
        logger.info(f"🖼️ Analyzing Page {img_info['page']}, Type: {img_info['type']}")
        logger.info(f"{'='*80}")
        
        # Save image
        img_path = output_dir / f"page_{img_info['page']:03d}_{img_info['type']}_{img_info['index']}.png"
        with open(img_path, 'wb') as f:
            f.write(img_info['data'])
        
        visual_result = {
            "page": img_info['page'],
            "type": img_info['type'],
            "path": str(img_path),
            "base64": base64.b64encode(img_info['data']).decode('utf-8'),
            "analyses": []
        }
        
        # Test each model
        for model_name, client in clients.items():
            logger.info(f"\n🤖 Testing {model_name}...")
            
            try:
                start = time.time()
                
                # Special handling for diagrams
                element_type = VisualElementType.DIAGRAM if img_info['page'] == 17 else None
                analysis_focus = "diagram_analysis" if img_info['page'] == 17 else "comprehensive"
                
                result = client.analyze_visual(
                    image_data=img_info['data'],
                    element_type=element_type,
                    analysis_focus=analysis_focus
                )
                elapsed = time.time() - start
                
                analysis = {
                    "model": model_name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "description": result.description,
                    "ocr_text": result.ocr_text,
                    "extracted_data": result.extracted_data,
                    "inference_time": elapsed,
                    "error_message": result.error_message
                }
                
                if result.success:
                    logger.info(f"✅ Success: {result.confidence:.0%} confidence")
                    logger.info(f"📝 Description preview: {result.description[:200]}...")
                    if img_info['page'] == 17:
                        logger.info(f"📊 DIAGRAM ANALYSIS: {result.description}")
                else:
                    logger.info(f"❌ Failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                analysis = {
                    "model": model_name,
                    "success": False,
                    "error_message": str(e),
                    "confidence": 0,
                    "inference_time": time.time() - start
                }
            
            visual_result["analyses"].append(analysis)
            
            # Don't cleanup after each image - keep models loaded
        
        results.append(visual_result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"bmw_x5_vlm_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_file}")
    
    # Generate HTML report
    generate_simple_html_report(results, output_dir / f"bmw_x5_vlm_comparison_{timestamp}.html")
    
    return results

def generate_simple_html_report(results, output_path):
    """Generate a simple HTML report focusing on key pages"""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW X5 G05 VLM Analysis - Key Pages</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .page-section {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .page-title {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #007bff;
        }}
        .highlight-diagram {{
            background-color: #fff3cd;
            border: 2px solid #ffc107;
        }}
        .image-preview {{
            max-width: 400px;
            margin: 20px auto;
            text-align: center;
        }}
        .image-preview img {{
            width: 100%;
            border: 1px solid #ddd;
            cursor: pointer;
        }}
        .model-analysis {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }}
        .model-name {{
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .description {{
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-left: 3px solid #007bff;
        }}
        .ocr-text {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f0f8ff;
            border-left: 3px solid #17a2b8;
            font-family: monospace;
            white-space: pre-wrap;
        }}
        .diagram-section {{
            background-color: #fff3cd;
            border: 3px solid #ffc107;
            padding: 20px;
            margin-top: 30px;
        }}
        .diagram-title {{
            font-size: 28px;
            color: #ff6b00;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BMW X5 G05 - VLM Analysis Report</h1>
        <p>Focus on Key Pages: Title, Tables, and Diagram</p>
        <p>Generated: {timestamp}</p>
    </div>
    
    {content}
    
    <script>
        function showImage(src) {{
            window.open(src, '_blank');
        }}
    </script>
</body>
</html>'''
    
    content = ""
    
    for result in results:
        page_class = "page-section"
        if result['page'] == 17:
            page_class += " diagram-section"
        
        content += f'''
        <div class="{page_class}">
            <div class="page-title">
                Page {result['page']} - {result['type'].replace('_', ' ').title()}
                {' ⚠️ DIAGRAM PAGE' if result['page'] == 17 else ''}
            </div>
            
            <div class="image-preview">
                <img src="data:image/png;base64,{result['base64']}" 
                     onclick="showImage(this.src)" 
                     alt="Page {result['page']}">
            </div>
        '''
        
        for analysis in result['analyses']:
            content += f'''
            <div class="model-analysis">
                <div class="model-name">{analysis['model']}</div>
                <div class="{('success' if analysis['success'] else 'failure')}">
                    {('✅ Success' if analysis['success'] else '❌ Failed')}
                    {f' - {analysis["confidence"]*100:.0f}% confidence' if analysis['success'] else ''}
                    <span style="color: #666; font-size: 0.9em;">({analysis['inference_time']:.1f}s)</span>
                </div>
                
                {f'''
                <div class="description">
                    <strong>Description:</strong><br>
                    {analysis['description']}
                </div>
                ''' if analysis['success'] else f'''
                <div class="failure">
                    Error: {analysis['error_message']}
                </div>
                '''}
                
                {f'''
                <div class="ocr-text">
                    <strong>OCR Text:</strong><br>
                    {analysis['ocr_text']}
                </div>
                ''' if analysis.get('ocr_text') else ''}
            </div>
            '''
        
        content += '</div>'
    
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content=content
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report generated: {output_path}")

if __name__ == "__main__":
    # Extract images from PDF
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    logger.info(f"📄 Extracting images from {pdf_path.name}...")
    
    images = extract_images_from_pdf(pdf_path)
    logger.info(f"📸 Extracted {len(images)} images/pages")
    
    # Analyze with VLMs
    analyze_images_with_vlms(images)