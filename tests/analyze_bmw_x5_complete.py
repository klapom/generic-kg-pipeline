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

def extract_all_pages_from_pdf(pdf_path):
    """Extract all pages from PDF as images"""
    pages = []
    
    doc = pymupdf.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        logger.info(f"Processing page {page_num + 1}/{len(doc)}")
        
        # Render the page as an image with high quality
        mat = pymupdf.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        pages.append({
            'page': page_num + 1,
            'data': img_data,
            'type': 'full_page'
        })
        
        pix = None
    
    doc.close()
    return pages

def analyze_all_pages_with_vlms(pages):
    """Analyze all pages with all VLMs"""
    
    output_dir = Path("tests/debugging/bmw_x5_complete_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize VLM clients
    logger.info("Initializing VLM clients...")
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
    total_pages = len(pages)
    
    for page_idx, page_info in enumerate(pages):
        logger.info(f"\n{'='*80}")
        logger.info(f"🖼️ Analyzing Page {page_info['page']}/{total_pages}")
        logger.info(f"{'='*80}")
        
        # Save page image
        img_path = output_dir / f"page_{page_info['page']:03d}.png"
        with open(img_path, 'wb') as f:
            f.write(page_info['data'])
        
        page_result = {
            "page": page_info['page'],
            "type": page_info['type'],
            "path": str(img_path),
            "base64": base64.b64encode(page_info['data']).decode('utf-8'),
            "analyses": []
        }
        
        # Determine if this page might contain special content
        special_type = None
        special_focus = "comprehensive"
        
        if page_info['page'] == 1:
            special_focus = "title_page"
        elif page_info['page'] in [11, 15]:
            special_type = VisualElementType.TABLE
            special_focus = "table_analysis"
        elif page_info['page'] == 17:
            special_type = VisualElementType.DIAGRAM
            special_focus = "diagram_analysis"
        
        # Test each model
        for model_name, client in clients.items():
            logger.info(f"\n🤖 Testing {model_name}...")
            
            try:
                start = time.time()
                
                result = client.analyze_visual(
                    image_data=page_info['data'],
                    element_type=special_type,
                    analysis_focus=special_focus
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
                    logger.info(f"✅ Success: {result.confidence:.0%} confidence, {elapsed:.1f}s")
                    logger.info(f"📝 Description preview: {result.description[:150]}...")
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
            
            page_result["analyses"].append(analysis)
        
        results.append(page_result)
        
        # Save intermediate results after each page
        intermediate_file = output_dir / "intermediate_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Page {page_info['page']} analysis complete")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"bmw_x5_complete_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ All pages analyzed! Results saved to: {results_file}")
    
    # Generate comprehensive HTML report
    generate_comprehensive_html_report(results, output_dir / f"bmw_x5_complete_comparison_{timestamp}.html")
    
    # Cleanup models
    for client in clients.values():
        if hasattr(client, "cleanup"):
            try:
                client.cleanup()
            except:
                pass
    
    return results

def generate_comprehensive_html_report(results, output_path):
    """Generate comprehensive HTML report for all pages"""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW X5 G05 Complete VLM Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #1a1a1a;
            color: white;
            padding: 30px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.8;
        }}
        .navigation {{
            background-color: #333;
            padding: 10px;
            text-align: center;
            position: sticky;
            top: 140px;
            z-index: 999;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .navigation button {{
            margin: 0 5px;
            padding: 8px 15px;
            border: none;
            background-color: #007bff;
            color: white;
            border-radius: 4px;
            cursor: pointer;
        }}
        .navigation button:hover {{
            background-color: #0056b3;
        }}
        .summary {{
            background-color: white;
            margin: 20px;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .container {{
            margin: 20px;
        }}
        .page-section {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .page-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .page-title {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .page-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .badge-diagram {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
        }}
        .badge-table {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #28a745;
        }}
        .badge-title {{
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #17a2b8;
        }}
        .image-preview {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-preview img {{
            max-width: 500px;
            width: 100%;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.3s;
        }}
        .image-preview img:hover {{
            transform: scale(1.05);
        }}
        .model-comparisons {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .model-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .model-name {{
            font-weight: bold;
            color: #007bff;
            font-size: 1.1em;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .failure {{
            color: #dc3545;
            font-weight: bold;
        }}
        .confidence-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .confidence-high {{
            background-color: #28a745;
            color: white;
        }}
        .confidence-medium {{
            background-color: #ffc107;
            color: #333;
        }}
        .confidence-low {{
            background-color: #dc3545;
            color: white;
        }}
        .inference-time {{
            color: #666;
            font-size: 0.9em;
        }}
        .description {{
            margin-top: 15px;
            padding: 15px;
            background-color: white;
            border-left: 3px solid #007bff;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .ocr-text {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f0f8ff;
            border-left: 3px solid #17a2b8;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
        }}
        .extracted-data {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f0fff0;
            border-left: 3px solid #28a745;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: 2% auto;
            max-width: 90%;
            max-height: 90%;
            text-align: center;
        }}
        .modal-content img {{
            max-width: 100%;
            max-height: 90vh;
        }}
        .close {{
            position: absolute;
            right: 35px;
            top: 15px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #bbb;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .comparison-table th, .comparison-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .comparison-table th {{
            background-color: #007bff;
            color: white;
        }}
        .comparison-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight-best {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BMW X5 G05 - Complete VLM Analysis</h1>
        <p>Comprehensive analysis of all {total_pages} pages</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="navigation">
        <button onclick="scrollToSection('summary')">Summary</button>
        <button onclick="scrollToSection('page1')">Title Page</button>
        <button onclick="scrollToSection('tables')">Tables</button>
        <button onclick="scrollToSection('diagrams')">Diagrams</button>
        <button onclick="scrollToSection('all-pages')">All Pages</button>
    </div>

    <div class="summary" id="summary">
        <h2>Analysis Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Pages</h3>
                <div class="value">{total_pages}</div>
            </div>
            <div class="stat-card">
                <h3>Total Analyses</h3>
                <div class="value">{total_analyses}</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="value">{success_rate}%</div>
            </div>
            <div class="stat-card">
                <h3>Avg. Confidence</h3>
                <div class="value">{avg_confidence}%</div>
            </div>
        </div>
        
        <h3 style="margin-top: 30px;">Model Performance Comparison</h3>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Success Rate</th>
                    <th>Avg. Confidence</th>
                    <th>Avg. Time (s)</th>
                    <th>Best For</th>
                </tr>
            </thead>
            <tbody>
                {model_comparison_rows}
            </tbody>
        </table>
    </div>

    <div class="container">
        {content}
    </div>

    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <img id="modalImage" src="">
        </div>
    </div>

    <script>
        function showModal(src) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}

        function scrollToSection(sectionId) {{
            document.getElementById(sectionId).scrollIntoView({{ behavior: 'smooth' }});
        }}

        window.onclick = function(event) {{
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}
    </script>
</body>
</html>'''
    
    # Calculate summary statistics
    total_pages = len(results)
    total_analyses = total_pages * 3  # 3 models
    successful_analyses = sum(1 for r in results for a in r['analyses'] if a['success'])
    success_rate = int((successful_analyses / total_analyses) * 100)
    
    # Calculate average confidence
    confidences = [a['confidence'] for r in results for a in r['analyses'] if a['success']]
    avg_confidence = int(sum(confidences) * 100 / len(confidences)) if confidences else 0
    
    # Calculate model statistics
    model_stats = {}
    for result in results:
        for analysis in result['analyses']:
            model = analysis['model']
            if model not in model_stats:
                model_stats[model] = {
                    'total': 0,
                    'success': 0,
                    'confidences': [],
                    'times': []
                }
            model_stats[model]['total'] += 1
            if analysis['success']:
                model_stats[model]['success'] += 1
                model_stats[model]['confidences'].append(analysis['confidence'])
            model_stats[model]['times'].append(analysis['inference_time'])
    
    # Generate model comparison rows
    model_comparison_rows = ""
    for model, stats in model_stats.items():
        success_rate = int((stats['success'] / stats['total']) * 100) if stats['total'] > 0 else 0
        avg_conf = int(sum(stats['confidences']) * 100 / len(stats['confidences'])) if stats['confidences'] else 0
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        
        best_for = ""
        if "Qwen" in model:
            best_for = "Fast analysis, good balance"
        elif "LLaVA" in model:
            best_for = "Detailed descriptions"
        elif "Pixtral" in model:
            best_for = "Technical diagrams, precise values"
        
        model_comparison_rows += f'''
        <tr>
            <td>{model}</td>
            <td class="{'highlight-best' if success_rate >= 90 else ''}">{success_rate}%</td>
            <td class="{'highlight-best' if avg_conf >= 85 else ''}">{avg_conf}%</td>
            <td class="{'highlight-best' if avg_time < 10 else ''}">{avg_time:.1f}</td>
            <td>{best_for}</td>
        </tr>
        '''
    
    # Generate content for each page
    content = ""
    
    # Group special pages
    title_pages = [r for r in results if r['page'] == 1]
    table_pages = [r for r in results if r['page'] in [11, 15]]
    diagram_pages = [r for r in results if r['page'] == 17]
    
    # Title page section
    if title_pages:
        content += '<div id="page1"><h2>Title Page Analysis</h2>'
        for result in title_pages:
            content += generate_page_section(result, 'badge-title')
        content += '</div>'
    
    # Tables section
    if table_pages:
        content += '<div id="tables"><h2>Table Pages Analysis</h2>'
        for result in table_pages:
            content += generate_page_section(result, 'badge-table')
        content += '</div>'
    
    # Diagrams section
    if diagram_pages:
        content += '<div id="diagrams"><h2>Diagram Pages Analysis</h2>'
        for result in diagram_pages:
            content += generate_page_section(result, 'badge-diagram')
        content += '</div>'
    
    # All pages section
    content += '<div id="all-pages"><h2>All Pages Analysis</h2>'
    for result in results:
        badge_class = ''
        if result['page'] == 1:
            badge_class = 'badge-title'
        elif result['page'] in [11, 15]:
            badge_class = 'badge-table'
        elif result['page'] == 17:
            badge_class = 'badge-diagram'
        content += generate_page_section(result, badge_class)
    content += '</div>'
    
    # Generate final HTML
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_pages=total_pages,
        total_analyses=total_analyses,
        success_rate=success_rate,
        avg_confidence=avg_confidence,
        model_comparison_rows=model_comparison_rows,
        content=content
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 Comprehensive HTML report generated: {output_path}")

def generate_page_section(result, badge_class=''):
    """Generate HTML section for a single page"""
    
    page_type = ""
    if result['page'] == 1:
        page_type = "Title Page"
    elif result['page'] in [11, 15]:
        page_type = "Table"
    elif result['page'] == 17:
        page_type = "Diagram"
    
    section_html = f'''
    <div class="page-section">
        <div class="page-header">
            <div>
                <span class="page-title">Page {result['page']}</span>
                {f'<span class="page-badge {badge_class}">{page_type}</span>' if page_type else ''}
            </div>
        </div>
        
        <div class="image-preview">
            <img src="data:image/png;base64,{result['base64']}" 
                 onclick="showModal(this.src)" 
                 alt="Page {result['page']}">
        </div>
        
        <div class="model-comparisons">
    '''
    
    for analysis in result['analyses']:
        confidence_class = 'confidence-high'
        if analysis['success']:
            if analysis['confidence'] < 0.7:
                confidence_class = 'confidence-low'
            elif analysis['confidence'] < 0.85:
                confidence_class = 'confidence-medium'
        
        section_html += f'''
        <div class="model-card">
            <div class="model-header">
                <span class="model-name">{analysis['model']}</span>
                <div>
                    {f'<span class="success">✅ Success</span>' if analysis['success'] else '<span class="failure">❌ Failed</span>'}
                    {f'<span class="confidence-badge {confidence_class}">{int(analysis["confidence"]*100)}%</span>' if analysis['success'] else ''}
                    <span class="inference-time">({analysis['inference_time']:.1f}s)</span>
                </div>
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
            
            {f'''
            <div class="extracted-data">
                <strong>Extracted Data:</strong><br>
                {json.dumps(analysis['extracted_data'], indent=2)}
            </div>
            ''' if analysis.get('extracted_data') else ''}
        </div>
        '''
    
    section_html += '''
        </div>
    </div>
    '''
    
    return section_html

if __name__ == "__main__":
    # Extract all pages from PDF
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    logger.info(f"📄 Extracting all pages from {pdf_path.name}...")
    
    pages = extract_all_pages_from_pdf(pdf_path)
    logger.info(f"📸 Extracted {len(pages)} pages")
    
    # Analyze all pages with VLMs
    analyze_all_pages_with_vlms(pages)