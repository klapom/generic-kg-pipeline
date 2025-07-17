#!/usr/bin/env python3
"""
Quick BMW Document VLM Demo

Analyzes only the most representative pages from the BMW document
for a quick but comprehensive VLM comparison.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF
import base64

# Setup logging
log_dir = Path("tests/debugging/quick_bmw_vlm_demo")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'demo_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, '.')

from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType


# Key pages to analyze - carefully selected for visual content
KEY_PAGES = [
    0,   # Title page
    2,   # First content page with images
    4,   # Technical diagrams
    8,   # Interior views
    11,  # Feature highlights
]


def extract_demo_pages(pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract only key demo pages"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"📄 Extracting {len(KEY_PAGES)} key pages from BMW document")
    doc = fitz.open(pdf_path)
    
    extracted = []
    
    for page_num in KEY_PAGES:
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        
        # Get page as high-quality image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for quality
        pix = page.get_pixmap(matrix=mat)
        
        img_path = output_dir / f"bmw_page_{page_num + 1:02d}.png"
        pix.save(img_path)
        
        # Get page description
        text_preview = page.get_text()[:100].strip()
        if not text_preview:
            text_preview = f"Visual content from page {page_num + 1}"
        
        extracted.append({
            "path": img_path,
            "page": page_num + 1,
            "description": f"Page {page_num + 1}: {text_preview}...",
            "width": pix.width,
            "height": pix.height
        })
        
        logger.info(f"  ✅ Page {page_num + 1} extracted")
    
    doc.close()
    return extracted


def analyze_with_all_vlms(image_path: Path, clients: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze one image with all VLMs"""
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    results = {}
    
    for model_name, client in clients.items():
        logger.info(f"    🤖 {model_name}...")
        
        try:
            start_time = time.time()
            
            result = client.analyze_visual(
                image_data=image_data,
                element_type=VisualElementType.IMAGE,
                analysis_focus="comprehensive"
            )
            
            elapsed = time.time() - start_time
            
            results[model_name] = {
                "success": result.success,
                "confidence": result.confidence,
                "description": result.description,
                "ocr_text": result.ocr_text,
                "extracted_data": result.extracted_data,
                "time": elapsed,
                "error": result.error_message
            }
            
            if result.success:
                logger.info(f"       ✅ {result.confidence:.0%} confidence in {elapsed:.1f}s")
            else:
                logger.info(f"       ❌ Failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"       ❌ Error: {e}")
            results[model_name] = {
                "success": False,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    return results


def generate_demo_html(all_results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate beautiful demo HTML report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🚗 BMW Document - VLM Analysis Demo</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0a0e27;
            color: #fff;
            overflow-x: hidden;
        }
        
        .hero {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 80px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: move 20s linear infinite;
        }
        
        @keyframes move {
            to { transform: translate(50px, 50px); }
        }
        
        h1 { 
            font-size: 4em;
            font-weight: 300;
            letter-spacing: -2px;
            margin-bottom: 20px;
            position: relative;
            z-index: 1;
        }
        
        .subtitle {
            font-size: 1.5em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: -40px auto 40px;
            max-width: 1000px;
            position: relative;
            z-index: 10;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.15);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .stat-number {
            font-size: 3em;
            font-weight: 700;
            color: #4facfe;
            display: block;
        }
        
        .stat-label {
            margin-top: 10px;
            opacity: 0.8;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }
        
        .page-analysis {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 40px;
            transition: all 0.3s;
        }
        
        .page-analysis:hover {
            background: rgba(255,255,255,0.08);
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .page-header {
            background: rgba(255,255,255,0.1);
            padding: 25px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .page-title {
            font-size: 1.5em;
            font-weight: 300;
        }
        
        .page-number {
            background: #4facfe;
            color: #0a0e27;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 600;
        }
        
        .page-content {
            display: grid;
            grid-template-columns: 500px 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        @media (max-width: 1200px) {
            .page-content {
                grid-template-columns: 1fr;
            }
        }
        
        .image-box {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            cursor: zoom-in;
        }
        
        .image-box img {
            width: 100%;
            height: auto;
            display: block;
            transition: transform 0.3s;
        }
        
        .image-box:hover img {
            transform: scale(1.05);
        }
        
        .models-comparison {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .model-result {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s;
        }
        
        .model-result:hover {
            background: rgba(255,255,255,0.08);
            border-color: rgba(255,255,255,0.2);
        }
        
        .model-result.qwen { border-left: 4px solid #667eea; }
        .model-result.llava { border-left: 4px solid #f093fb; }
        .model-result.pixtral { border-left: 4px solid #4facfe; }
        
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .model-name {
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .model-badges {
            display: flex;
            gap: 10px;
        }
        
        .badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .badge.confidence {
            background: rgba(76, 175, 254, 0.2);
            color: #4facfe;
        }
        
        .badge.time {
            background: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.8);
        }
        
        .description {
            line-height: 1.8;
            opacity: 0.9;
            margin-bottom: 15px;
        }
        
        .data-box {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            font-family: 'SF Mono', 'Monaco', monospace;
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .evaluation {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .stars {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        
        .star {
            font-size: 28px;
            color: rgba(255,255,255,0.2);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .star:hover {
            color: #ffd700;
            transform: scale(1.2);
        }
        
        .star.active {
            color: #ffd700;
        }
        
        .summary {
            background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(102, 126, 234, 0.1));
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-top: 60px;
        }
        
        .export-btn {
            background: linear-gradient(135deg, #4facfe, #667eea);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 30px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(79, 172, 254, 0.3);
        }
        
        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(79, 172, 254, 0.4);
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.95);
            cursor: zoom-out;
        }
        
        .modal img {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="hero">
        <h1>🚗 BMW Document Analysis</h1>
        <div class="subtitle">Visual Language Model Performance Comparison</div>
    </div>
    
    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <span class="stat-number">""" + str(len(all_results)) + """</span>
                <span class="stat-label">Pages Analyzed</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">3</span>
                <span class="stat-label">VLM Models</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="avgConfidence">-</span>
                <span class="stat-label">Avg Confidence</span>
            </div>
            <div class="stat-card">
                <span class="stat-number" id="totalTime">-</span>
                <span class="stat-label">Total Time</span>
            </div>
        </div>
"""
    
    # Add results for each page
    for page_result in all_results:
        html_content += f"""
        <div class="page-analysis">
            <div class="page-header">
                <h2 class="page-title">{page_result['description']}</h2>
                <span class="page-number">Page {page_result['page']}</span>
            </div>
            
            <div class="page-content">
                <div class="image-box" onclick="openModal('{page_result['base64']}')">
                    <img src="data:image/png;base64,{page_result['base64']}" alt="Page {page_result['page']}">
                </div>
                
                <div class="models-comparison">
"""
        
        # Add each model's analysis
        model_classes = {
            'Qwen2.5-VL-7B': 'qwen',
            'LLaVA-1.6-Mistral-7B': 'llava',
            'Pixtral-12B': 'pixtral'
        }
        
        for model_name, analysis in page_result['analyses'].items():
            model_class = model_classes.get(model_name, '')
            
            if analysis['success']:
                confidence = int(analysis['confidence'] * 100)
                html_content += f"""
                    <div class="model-result {model_class}">
                        <div class="model-header">
                            <div class="model-name">{model_name}</div>
                            <div class="model-badges">
                                <span class="badge confidence">{confidence}%</span>
                                <span class="badge time">⏱️ {analysis['time']:.1f}s</span>
                            </div>
                        </div>
                        
                        <div class="description">{analysis['description']}</div>
"""
                
                if analysis.get('ocr_text'):
                    html_content += f"""
                        <div class="data-box">
                            <strong>🔤 OCR Text:</strong><br>
                            {analysis['ocr_text']}
                        </div>
"""
                
                if analysis.get('extracted_data'):
                    html_content += f"""
                        <div class="data-box">
                            <strong>📊 Extracted Data:</strong><br>
                            <pre>{json.dumps(analysis['extracted_data'], indent=2)}</pre>
                        </div>
"""
                
                # Add evaluation
                eval_id = f"page{page_result['page']}_{model_name}"
                html_content += f"""
                        <div class="evaluation">
                            <strong>Rate this analysis:</strong>
                            <div class="stars" id="stars_{eval_id}">
                                {"".join([f'<span class="star" onclick="rate(\'{eval_id}\', {i+1})">★</span>' for i in range(5)])}
                            </div>
                        </div>
                    </div>
"""
            else:
                html_content += f"""
                    <div class="model-result {model_class}">
                        <div class="model-header">
                            <div class="model-name">{model_name}</div>
                            <div class="model-badges">
                                <span class="badge" style="background: rgba(255,0,0,0.2); color: #ff6b6b;">Failed</span>
                            </div>
                        </div>
                        <div style="opacity: 0.7;">❌ {analysis.get('error', 'Unknown error')}</div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <div class="summary">
            <h2 style="font-size: 2em; margin-bottom: 20px;">📊 Analysis Complete!</h2>
            <p style="font-size: 1.2em; opacity: 0.9; margin-bottom: 30px;">
                Export your evaluations and detailed analysis results
            </p>
            <button class="export-btn" onclick="exportResults()">
                Export Results & Evaluations
            </button>
        </div>
    </div>
    
    <div class="modal" id="modal" onclick="closeModal()">
        <img id="modalImg" src="">
    </div>
    
    <script>
        const evaluations = {};
        const results = """ + json.dumps(all_results) + """;
        
        // Calculate stats
        let totalConf = 0;
        let confCount = 0;
        let totalTime = 0;
        
        results.forEach(page => {
            Object.values(page.analyses).forEach(analysis => {
                if (analysis.success) {
                    totalConf += analysis.confidence;
                    confCount++;
                }
                totalTime += analysis.time || 0;
            });
        });
        
        document.getElementById('avgConfidence').textContent = 
            confCount > 0 ? Math.round(totalConf / confCount * 100) + '%' : 'N/A';
        document.getElementById('totalTime').textContent = 
            Math.round(totalTime) + 's';
        
        function rate(evalId, rating) {
            evaluations[evalId] = rating;
            
            // Update display
            const stars = document.querySelectorAll(`#stars_${evalId} .star`);
            stars.forEach((star, idx) => {
                star.classList.toggle('active', idx < rating);
            });
        }
        
        function openModal(base64) {
            const modal = document.getElementById('modal');
            const img = document.getElementById('modalImg');
            modal.style.display = 'block';
            img.src = 'data:image/png;base64,' + base64;
        }
        
        function closeModal() {
            document.getElementById('modal').style.display = 'none';
        }
        
        function exportResults() {
            const exportData = {
                timestamp: new Date().toISOString(),
                document: 'BMW 3er G20 Preview',
                pages_analyzed: results.length,
                evaluations: evaluations,
                detailed_results: results,
                summary: {
                    avg_confidence: confCount > 0 ? (totalConf / confCount) : 0,
                    total_time: totalTime,
                    models_compared: ['Qwen2.5-VL-7B', 'LLaVA-1.6-Mistral-7B', 'Pixtral-12B']
                }
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                                {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bmw_vlm_analysis_${new Date().toISOString().slice(0,10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('✅ Results exported successfully!');
        }
    </script>
</body>
</html>"""
    
    # Save report
    report_path = output_dir / "bmw_vlm_demo_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path


def main():
    """Run quick BMW VLM demo"""
    
    logger.info("🚀 Starting Quick BMW VLM Demo")
    logger.info("=" * 60)
    
    # Input PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Extract demo pages
    image_dir = log_dir / "demo_pages"
    pages = extract_demo_pages(pdf_path, image_dir)
    
    logger.info(f"\n📊 Analyzing {len(pages)} key pages with 3 VLMs...")
    
    # Initialize VLMs
    logger.info("\n🔧 Loading VLM models...")
    
    start_init = time.time()
    clients = {
        "Qwen2.5-VL-7B": TransformersQwen25VLClient(
            temperature=0.2,
            max_new_tokens=512
        ),
        "LLaVA-1.6-Mistral-7B": TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.2,
            max_new_tokens=512
        ),
        "Pixtral-12B": TransformersPixtralClient(
            temperature=0.2,
            max_new_tokens=512,
            load_in_8bit=True
        )
    }
    logger.info(f"✅ Models loaded in {time.time() - start_init:.1f}s")
    
    # Analyze pages
    all_results = []
    total_start = time.time()
    
    for page_info in pages:
        logger.info(f"\n📄 Analyzing page {page_info['page']}...")
        
        # Analyze with all models
        analyses = analyze_with_all_vlms(page_info['path'], clients)
        
        # Convert image to base64 for HTML
        with open(page_info['path'], 'rb') as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
        
        all_results.append({
            "page": page_info['page'],
            "description": page_info['description'],
            "base64": base64_img,
            "analyses": analyses
        })
    
    total_time = time.time() - total_start
    
    # Cleanup
    logger.info("\n🧹 Cleaning up resources...")
    for client in clients.values():
        if hasattr(client, 'cleanup'):
            try:
                client.cleanup()
            except:
                pass
    
    # Save results
    results_file = log_dir / "demo_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate report
    logger.info("\n📄 Generating demo report...")
    report_path = generate_demo_html(all_results, log_dir)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ DEMO COMPLETE")
    logger.info("="*60)
    logger.info(f"📄 Pages analyzed: {len(pages)}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s")
    logger.info(f"⚡ Average per page: {total_time/len(pages):.1f}s")
    
    # Model performance
    model_stats = {}
    for result in all_results:
        for model, analysis in result['analyses'].items():
            if model not in model_stats:
                model_stats[model] = {'success': 0, 'total': 0, 'times': []}
            
            model_stats[model]['total'] += 1
            if analysis['success']:
                model_stats[model]['success'] += 1
                model_stats[model]['times'].append(analysis['time'])
    
    logger.info("\n📊 Model Performance:")
    for model, stats in model_stats.items():
        success_rate = stats['success'] / stats['total'] * 100
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        logger.info(f"  • {model}: {success_rate:.0f}% success, {avg_time:.1f}s avg")
    
    logger.info(f"\n🌐 Öffnen Sie den Bericht:")
    logger.info(f"   {report_path}")
    
    return str(report_path)


if __name__ == "__main__":
    report_path = main()
    if report_path:
        print(f"\n✅ Demo abgeschlossen!")
        print(f"🌐 HTML Bericht: {report_path}")