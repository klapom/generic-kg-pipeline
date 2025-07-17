#!/usr/bin/env python3
"""
Key Pages VLM Analysis

Analyzes only key pages with images/diagrams from the BMW document
to provide a comprehensive but efficient evaluation.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image

# Setup logging
log_dir = Path("tests/debugging/key_pages_vlm_analysis")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'analysis_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, '.')

from core.utils.model_cache_manager import ModelCacheManager
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType


def identify_key_pages(pdf_path: Path) -> List[int]:
    """Identify pages with significant visual content"""
    doc = fitz.open(pdf_path)
    key_pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Check for images
        image_list = page.get_images()
        significant_images = [img for img in image_list if img[2] > 200 and img[3] > 200]  # width, height
        
        # Check text density (pages with less text often have diagrams)
        text = page.get_text()
        text_density = len(text) / (page.rect.width * page.rect.height)
        
        # Criteria for key pages
        if (len(significant_images) > 0 or 
            text_density < 0.002 or  # Low text density might indicate diagrams
            page_num < 5 or  # First few pages
            page_num == 0):  # Title page
            key_pages.append(page_num)
    
    doc.close()
    
    # Limit to most important pages
    return sorted(set(key_pages))[:15]  # Max 15 pages


def extract_page_visuals(pdf_path: Path, page_nums: List[int], output_dir: Path) -> List[Dict[str, Any]]:
    """Extract visuals from specific pages"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"📄 Extracting visuals from {len(page_nums)} key pages")
    doc = fitz.open(pdf_path)
    
    extracted_visuals = []
    
    for page_num in page_nums:
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        
        # Always save full page for context
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        page_img_path = output_dir / f"page_{page_num + 1:03d}_full.png"
        pix.save(page_img_path)
        
        extracted_visuals.append({
            "path": page_img_path,
            "page": page_num + 1,
            "type": "full_page",
            "width": pix.width,
            "height": pix.height,
            "description": f"Page {page_num + 1} - Full view"
        })
        
        # Extract individual images if any
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Only significant images
                if pix.width < 300 or pix.height < 300:
                    continue
                
                # Convert to RGB if necessary
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Save image
                img_path = output_dir / f"page_{page_num + 1:03d}_img_{img_index + 1:02d}.png"
                pix.save(img_path)
                
                extracted_visuals.append({
                    "path": img_path,
                    "page": page_num + 1,
                    "type": "embedded_image",
                    "width": pix.width,
                    "height": pix.height,
                    "description": f"Page {page_num + 1} - Image {img_index + 1}"
                })
                
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
                continue
    
    doc.close()
    
    logger.info(f"✅ Extracted {len(extracted_visuals)} visuals")
    return extracted_visuals


def analyze_visual_batch(visuals: List[Dict[str, Any]], clients: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze a batch of visuals with all VLMs"""
    results = []
    
    for idx, visual_info in enumerate(visuals):
        logger.info(f"\n{'='*60}")
        logger.info(f"📸 Analyzing visual {idx + 1}/{len(visuals)}: {visual_info['description']}")
        
        # Load image once
        with open(visual_info['path'], 'rb') as f:
            image_data = f.read()
        
        # Prepare result
        visual_result = {
            "visual_id": f"vis_{idx:03d}",
            "path": str(visual_info['path']),
            "page": visual_info['page'],
            "type": visual_info['type'],
            "width": visual_info['width'],
            "height": visual_info['height'],
            "description": visual_info['description'],
            "base64": base64.b64encode(image_data).decode('utf-8'),
            "analyses": []
        }
        
        # Test each VLM
        for model_name, client in clients.items():
            logger.info(f"   🤖 {model_name}...")
            
            try:
                start_time = time.time()
                
                result = client.analyze_visual(
                    image_data=image_data,
                    element_type=VisualElementType.IMAGE,
                    analysis_focus="comprehensive"
                )
                
                inference_time = time.time() - start_time
                
                analysis = {
                    "model": model_name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "description": result.description,
                    "ocr_text": result.ocr_text,
                    "extracted_data": result.extracted_data,
                    "inference_time": inference_time,
                    "error_message": result.error_message
                }
                
                if result.success:
                    logger.info(f"      ✅ Confidence: {result.confidence:.0%}, Time: {inference_time:.1f}s")
                else:
                    logger.info(f"      ❌ Failed: {result.error_message}")
                
            except Exception as e:
                logger.error(f"      ❌ Error: {e}")
                analysis = {
                    "model": model_name,
                    "success": False,
                    "error_message": str(e),
                    "confidence": 0,
                    "inference_time": 0
                }
            
            visual_result['analyses'].append(analysis)
        
        results.append(visual_result)
    
    return results


def generate_interactive_report(all_results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate interactive HTML evaluation report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🎯 BMW Document - Key Pages VLM Analysis</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f5f7fa;
            color: #2c3e50;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        h1 { 
            margin: 0;
            font-size: 3em;
            font-weight: 300;
            letter-spacing: -1px;
        }
        .subtitle {
            margin-top: 10px;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s;
        }
        .summary-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.12);
        }
        .card-value {
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card-label {
            color: #7f8c8d;
            margin-top: 5px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .visual-section {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        .visual-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }
        .page-badge {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .visual-content {
            display: grid;
            grid-template-columns: 450px 1fr;
            gap: 30px;
            align-items: start;
        }
        @media (max-width: 1200px) {
            .visual-content {
                grid-template-columns: 1fr;
            }
        }
        .image-container {
            position: relative;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 12px;
        }
        .image-container img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            cursor: zoom-in;
            transition: transform 0.3s;
        }
        .image-container:hover img {
            transform: scale(1.02);
        }
        .zoom-hint {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .image-container:hover .zoom-hint {
            opacity: 1;
        }
        .analyses-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .model-analysis {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border: 2px solid transparent;
            transition: all 0.3s;
        }
        .model-analysis:hover {
            border-color: #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .model-analysis.qwen { border-left: 4px solid #667eea; }
        .model-analysis.llava { border-left: 4px solid #f093fb; }
        .model-analysis.pixtral { border-left: 4px solid #4facfe; }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .model-name {
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }
        .metrics {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .confidence-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .confidence-high { background: #d4edda; color: #155724; }
        .confidence-medium { background: #fff3cd; color: #856404; }
        .confidence-low { background: #f8d7da; color: #721c24; }
        .time-badge {
            color: #7f8c8d;
            font-size: 0.85em;
        }
        .description {
            line-height: 1.7;
            color: #495057;
            margin-bottom: 15px;
        }
        .data-box {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-family: 'SF Mono', 'Monaco', monospace;
            font-size: 0.9em;
            border: 1px solid #e0e0e0;
        }
        .evaluation-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #f0f0f0;
        }
        .rating-container {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .stars {
            display: flex;
            gap: 5px;
        }
        .star {
            font-size: 28px;
            color: #ddd;
            cursor: pointer;
            transition: all 0.2s;
        }
        .star:hover {
            transform: scale(1.2);
        }
        .star.active {
            color: #ffd700;
        }
        .notes-input {
            flex: 1;
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        .notes-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .export-section {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            text-align: center;
            margin-top: 40px;
        }
        .export-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 30px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.95);
            cursor: zoom-out;
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 2%;
            border-radius: 8px;
        }
        .close {
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: 300;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .close:hover {
            transform: rotate(90deg);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 BMW Document Analysis</h1>
        <div class="subtitle">Key Pages Visual Language Model Comparison</div>
    </div>
    
    <div class="container">
        <div class="summary-cards">
            <div class="summary-card">
                <div class="card-value" id="totalVisuals">0</div>
                <div class="card-label">Total Visuals</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="totalAnalyses">0</div>
                <div class="card-label">Total Analyses</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="avgConfidence">0%</div>
                <div class="card-label">Avg Confidence</div>
            </div>
            <div class="summary-card">
                <div class="card-value" id="totalTime">0s</div>
                <div class="card-label">Analysis Time</div>
            </div>
        </div>
        
        <div id="visualResults"></div>
        
        <div class="export-section">
            <h2>📊 Export Your Evaluation</h2>
            <p>Download all ratings and notes for further analysis</p>
            <button class="export-btn" onclick="exportEvaluations()">
                Export Evaluations
            </button>
        </div>
    </div>
    
    <!-- Modal for full-size images -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        // Data storage
        const allResults = """ + json.dumps(all_results) + """;
        const evaluations = {};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            displayResults();
            updateSummary();
        });
        
        function displayResults() {
            const container = document.getElementById('visualResults');
            
            allResults.forEach((visual, idx) => {
                const section = createVisualSection(visual, idx);
                container.appendChild(section);
            });
        }
        
        function createVisualSection(visual, idx) {
            const section = document.createElement('div');
            section.className = 'visual-section';
            
            // Header
            const header = document.createElement('div');
            header.className = 'visual-header';
            header.innerHTML = `
                <h2>${visual.description}</h2>
                <span class="page-badge">Page ${visual.page}</span>
            `;
            section.appendChild(header);
            
            // Content
            const content = document.createElement('div');
            content.className = 'visual-content';
            
            // Image
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';
            imageContainer.innerHTML = `
                <img src="data:image/png;base64,${visual.base64}" 
                     onclick="openModal(this.src)" 
                     alt="${visual.description}">
                <div class="zoom-hint">Click to zoom</div>
            `;
            content.appendChild(imageContainer);
            
            // Analyses
            const analysesContainer = document.createElement('div');
            analysesContainer.className = 'analyses-container';
            
            visual.analyses.forEach(analysis => {
                const modelDiv = createModelAnalysis(analysis, visual.visual_id);
                analysesContainer.appendChild(modelDiv);
            });
            
            content.appendChild(analysesContainer);
            section.appendChild(content);
            
            return section;
        }
        
        function createModelAnalysis(analysis, visualId) {
            const modelClass = {
                'Qwen2.5-VL-7B': 'qwen',
                'LLaVA-1.6-Mistral-7B': 'llava',
                'Pixtral-12B': 'pixtral'
            };
            
            const div = document.createElement('div');
            div.className = 'model-analysis ' + (modelClass[analysis.model] || '');
            
            const confidence = (analysis.confidence || 0) * 100;
            const confClass = confidence > 80 ? 'high' : (confidence > 60 ? 'medium' : 'low');
            
            let content = `
                <div class="model-header">
                    <div class="model-name">${analysis.model}</div>
                    <div class="metrics">
                        <span class="confidence-badge confidence-${confClass}">
                            ${confidence.toFixed(0)}% confidence
                        </span>
                        <span class="time-badge">⏱️ ${analysis.inference_time.toFixed(1)}s</span>
                    </div>
                </div>
            `;
            
            if (analysis.success) {
                content += `<div class="description">${analysis.description || 'No description'}</div>`;
                
                if (analysis.ocr_text) {
                    content += `
                        <div class="data-box">
                            <strong>🔤 OCR Text:</strong><br>
                            ${analysis.ocr_text}
                        </div>
                    `;
                }
                
                if (analysis.extracted_data) {
                    content += `
                        <div class="data-box">
                            <strong>📊 Extracted Data:</strong><br>
                            <pre>${JSON.stringify(analysis.extracted_data, null, 2)}</pre>
                        </div>
                    `;
                }
                
                // Evaluation
                const evalId = `${visualId}_${analysis.model}`;
                content += `
                    <div class="evaluation-section">
                        <div class="rating-container">
                            <div class="stars" id="stars_${evalId}">
                                ${[1,2,3,4,5].map(i => 
                                    `<span class="star" onclick="setRating('${evalId}', ${i})">★</span>`
                                ).join('')}
                            </div>
                            <input type="text" 
                                   class="notes-input" 
                                   id="notes_${evalId}"
                                   placeholder="Add notes about this analysis..."
                                   onchange="updateNotes('${evalId}', this.value)">
                        </div>
                    </div>
                `;
            } else {
                content += `
                    <div style="color: #e74c3c; font-style: italic;">
                        ❌ ${analysis.error_message || 'Analysis failed'}
                    </div>
                `;
            }
            
            div.innerHTML = content;
            return div;
        }
        
        function setRating(evalId, rating) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].rating = rating;
            
            // Update display
            const stars = document.querySelectorAll(`#stars_${evalId} .star`);
            stars.forEach((star, idx) => {
                star.classList.toggle('active', idx < rating);
            });
            
            updateSummary();
        }
        
        function updateNotes(evalId, notes) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].notes = notes;
        }
        
        function updateSummary() {
            // Calculate statistics
            let totalAnalyses = 0;
            let totalConfidence = 0;
            let successfulAnalyses = 0;
            let totalTime = 0;
            
            allResults.forEach(visual => {
                visual.analyses.forEach(analysis => {
                    totalAnalyses++;
                    totalTime += analysis.inference_time || 0;
                    
                    if (analysis.success) {
                        successfulAnalyses++;
                        totalConfidence += analysis.confidence || 0;
                    }
                });
            });
            
            // Update display
            document.getElementById('totalVisuals').textContent = allResults.length;
            document.getElementById('totalAnalyses').textContent = totalAnalyses;
            document.getElementById('avgConfidence').textContent = 
                successfulAnalyses > 0 ? 
                Math.round(totalConfidence / successfulAnalyses * 100) + '%' : '0%';
            document.getElementById('totalTime').textContent = Math.round(totalTime) + 's';
        }
        
        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            modal.style.display = 'block';
            modalImg.src = src;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        function exportEvaluations() {
            const exportData = {
                timestamp: new Date().toISOString(),
                document: 'Preview_BMW_3er_G20.pdf',
                evaluations: evaluations,
                summary: {
                    totalVisuals: allResults.length,
                    totalEvaluations: Object.keys(evaluations).length,
                    avgRating: calculateAvgRating()
                },
                detailedResults: []
            };
            
            // Add detailed results
            allResults.forEach(visual => {
                const visualEvals = {};
                visual.analyses.forEach(analysis => {
                    const evalId = `${visual.visual_id}_${analysis.model}`;
                    if (evaluations[evalId]) {
                        visualEvals[analysis.model] = {
                            ...evaluations[evalId],
                            confidence: analysis.confidence,
                            inferenceTime: analysis.inference_time
                        };
                    }
                });
                
                exportData.detailedResults.push({
                    visual: visual.description,
                    page: visual.page,
                    evaluations: visualEvals
                });
            });
            
            // Download
            const blob = new Blob([JSON.stringify(exportData, null, 2)], 
                                {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `bmw_vlm_evaluation_${new Date().toISOString().slice(0,10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('✅ Evaluations exported successfully!');
        }
        
        function calculateAvgRating() {
            const ratings = Object.values(evaluations)
                .filter(e => e.rating)
                .map(e => e.rating);
            
            return ratings.length > 0 ? 
                (ratings.reduce((a, b) => a + b, 0) / ratings.length).toFixed(1) : 0;
        }
    </script>
</body>
</html>"""
    
    # Save report
    report_path = output_dir / "key_pages_vlm_evaluation.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path


def main():
    """Run key pages analysis"""
    
    logger.info("🚀 Starting Key Pages VLM Analysis")
    logger.info("=" * 80)
    
    # Input PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Identify key pages
    logger.info("\n🔍 Identifying key pages with visual content...")
    key_pages = identify_key_pages(pdf_path)
    logger.info(f"✅ Found {len(key_pages)} key pages: {key_pages}")
    
    # Extract visuals
    image_dir = log_dir / "extracted_visuals"
    visuals = extract_page_visuals(pdf_path, key_pages, image_dir)
    
    if not visuals:
        logger.error("No visuals extracted!")
        return
    
    logger.info(f"\n📊 Will analyze {len(visuals)} visuals with 3 VLMs")
    
    # Initialize VLM clients
    logger.info("\n🔧 Initializing VLM clients...")
    
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
    
    # Analyze visuals
    total_start = time.time()
    results = analyze_visual_batch(visuals, clients)
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
    results_file = log_dir / "key_pages_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate report
    logger.info("\n📄 Generating interactive evaluation report...")
    report_path = generate_interactive_report(results, log_dir)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"📊 Visuals analyzed: {len(results)}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"⚡ Average per visual: {total_time/len(results):.1f}s")
    
    # Model summary
    model_stats = {}
    for result in results:
        for analysis in result['analyses']:
            model = analysis['model']
            if model not in model_stats:
                model_stats[model] = {'success': 0, 'total': 0, 'times': []}
            
            model_stats[model]['total'] += 1
            if analysis['success']:
                model_stats[model]['success'] += 1
                model_stats[model]['times'].append(analysis['inference_time'])
    
    logger.info("\n📈 Model Performance:")
    for model, stats in model_stats.items():
        success_rate = stats['success'] / stats['total'] * 100
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        
        logger.info(f"  • {model}: {success_rate:.0f}% success, {avg_time:.1f}s avg")
    
    logger.info(f"\n🌐 Open the evaluation report:")
    logger.info(f"   {report_path}")
    
    return {
        "report": str(report_path),
        "results": str(results_file),
        "visuals": str(image_dir),
        "analysis_time": total_time
    }


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Analyse abgeschlossen!")
        print(f"🌐 Öffnen Sie den Bericht: {results['report']}")