#!/usr/bin/env python3
"""
Full Document VLM Analysis

Analyzes all images/diagrams in the BMW document with all 3 VLMs
and generates an interactive HTML report for evaluation.
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
log_dir = Path("tests/debugging/full_document_vlm_analysis")
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


def extract_all_images_from_pdf(pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Extract all significant images from PDF"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"📄 Extracting images from {pdf_path.name}")
    doc = fitz.open(pdf_path)
    
    extracted_images = []
    total_images = 0
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        # Also get page as image for pages with complex layouts
        if page_num < 20:  # Focus on first 20 pages
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            if pix.width > 800 and pix.height > 600:  # Only save substantial pages
                page_img_path = output_dir / f"page_{page_num + 1:03d}_full.png"
                pix.save(page_img_path)
                
                extracted_images.append({
                    "path": page_img_path,
                    "page": page_num + 1,
                    "type": "full_page",
                    "width": pix.width,
                    "height": pix.height,
                    "description": f"Full page {page_num + 1}"
                })
        
        # Extract individual images
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Skip small images
                if pix.width < 200 or pix.height < 200:
                    continue
                
                # Convert to RGB if necessary
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Save image
                img_path = output_dir / f"page_{page_num + 1:03d}_img_{img_index + 1:02d}.png"
                pix.save(img_path)
                
                extracted_images.append({
                    "path": img_path,
                    "page": page_num + 1,
                    "type": "embedded_image",
                    "width": pix.width,
                    "height": pix.height,
                    "description": f"Image {img_index + 1} from page {page_num + 1}"
                })
                
                total_images += 1
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                continue
    
    doc.close()
    
    logger.info(f"✅ Extracted {len(extracted_images)} images/pages from document")
    return extracted_images


def analyze_image_with_vlm(client, image_path: Path, model_name: str) -> Dict[str, Any]:
    """Analyze a single image with a VLM"""
    
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        start_time = time.time()
        
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        inference_time = time.time() - start_time
        
        return {
            "model": model_name,
            "success": result.success,
            "confidence": result.confidence,
            "description": result.description,
            "ocr_text": result.ocr_text,
            "extracted_data": result.extracted_data,
            "inference_time": inference_time,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path} with {model_name}: {e}")
        return {
            "model": model_name,
            "success": False,
            "error_message": str(e),
            "confidence": 0,
            "inference_time": 0
        }


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_evaluation_html(all_results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate interactive HTML report for evaluation"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🔍 BMW Document - Complete VLM Analysis</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f0f2f5;
        }
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
            box-shadow: 0 2px 20px rgba(0,0,0,0.2);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        h1 { 
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .nav-stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .stat-item {
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        .filter-controls {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        select, input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .image-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            position: relative;
        }
        .image-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 15px;
        }
        .page-info {
            background: #e3f2fd;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            color: #1976d2;
        }
        .image-container {
            display: flex;
            gap: 30px;
            align-items: start;
        }
        .image-preview {
            flex: 0 0 400px;
            max-width: 400px;
        }
        .image-preview img {
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: zoom-in;
            transition: transform 0.3s;
        }
        .image-preview img:hover {
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .analysis-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .model-analysis {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ddd;
            position: relative;
        }
        .model-analysis.qwen { border-left-color: #667eea; }
        .model-analysis.llava { border-left-color: #f093fb; }
        .model-analysis.pixtral { border-left-color: #4facfe; }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .model-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        .model-metrics {
            display: flex;
            gap: 15px;
            font-size: 0.9em;
        }
        .confidence {
            padding: 4px 10px;
            border-radius: 15px;
            font-weight: bold;
        }
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
        .description {
            line-height: 1.6;
            color: #333;
            margin-bottom: 10px;
        }
        .ocr-section {
            background: #e8f5e9;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .data-section {
            background: #e3f2fd;
            padding: 12px;
            border-radius: 6px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .evaluation-controls {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .rating-stars {
            display: flex;
            gap: 5px;
        }
        .star {
            font-size: 24px;
            color: #ddd;
            cursor: pointer;
            transition: color 0.2s;
        }
        .star:hover, .star.active {
            color: #ffd700;
        }
        .notes-input {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            cursor: zoom-out;
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 2%;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .summary-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .export-btn {
            background: #4caf50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        .export-btn:hover {
            background: #45a049;
        }
        @media (max-width: 1200px) {
            .image-container {
                flex-direction: column;
            }
            .image-preview {
                flex: none;
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 BMW Document - Complete VLM Analysis</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Comprehensive analysis of all images and diagrams</p>
        <div class="nav-stats">
            <div class="stat-item">📄 Total Images: <span id="totalImages">0</span></div>
            <div class="stat-item">📊 Analyzed: <span id="analyzedImages">0</span></div>
            <div class="stat-item">⭐ Evaluated: <span id="evaluatedImages">0</span></div>
            <div class="stat-item">🕒 Analysis Time: <span id="totalTime">0</span>s</div>
        </div>
    </div>
    
    <div class="container">
        <div class="filter-controls">
            <div class="filter-group">
                <label>Page Range:</label>
                <input type="number" id="pageFrom" placeholder="From" min="1" style="width: 80px">
                <input type="number" id="pageTo" placeholder="To" min="1" style="width: 80px">
            </div>
            <div class="filter-group">
                <label>Image Type:</label>
                <select id="imageType">
                    <option value="all">All Types</option>
                    <option value="full_page">Full Pages</option>
                    <option value="embedded_image">Embedded Images</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Model:</label>
                <select id="modelFilter">
                    <option value="all">All Models</option>
                    <option value="Qwen2.5-VL-7B">Qwen2.5-VL</option>
                    <option value="LLaVA-1.6-Mistral-7B">LLaVA</option>
                    <option value="Pixtral-12B">Pixtral</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Min Confidence:</label>
                <input type="range" id="confidenceFilter" min="0" max="100" value="0" style="width: 150px">
                <span id="confidenceValue">0%</span>
            </div>
            <button onclick="applyFilters()">Apply Filters</button>
            <button onclick="resetFilters()">Reset</button>
        </div>
        
        <div class="summary-section">
            <h2>📊 Analysis Summary</h2>
            <div id="summaryContent"></div>
            <button class="export-btn" onclick="exportEvaluations()">📥 Export Evaluations</button>
        </div>
        
        <div id="imageResults"></div>
    </div>
    
    <!-- Modal for full-size images -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        // Store all results for filtering
        const allResults = """ + json.dumps(all_results) + """;
        
        // Store evaluations
        const evaluations = {};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            displayResults(allResults);
            updateStats();
            
            // Setup event listeners
            document.getElementById('confidenceFilter').addEventListener('input', function(e) {
                document.getElementById('confidenceValue').textContent = e.target.value + '%';
            });
        });
        
        function displayResults(results) {
            const container = document.getElementById('imageResults');
            container.innerHTML = '';
            
            results.forEach((imageResult, index) => {
                const section = createImageSection(imageResult, index);
                container.appendChild(section);
            });
        }
        
        function createImageSection(imageResult, index) {
            const section = document.createElement('div');
            section.className = 'image-section';
            section.dataset.page = imageResult.page;
            section.dataset.type = imageResult.type;
            
            // Header
            const header = document.createElement('div');
            header.className = 'image-header';
            header.innerHTML = `
                <h3>${imageResult.description}</h3>
                <div>
                    <span class="page-info">Page ${imageResult.page}</span>
                    <span class="page-info">${imageResult.width}×${imageResult.height}</span>
                    <span class="page-info">${imageResult.type.replace('_', ' ')}</span>
                </div>
            `;
            section.appendChild(header);
            
            // Content container
            const container = document.createElement('div');
            container.className = 'image-container';
            
            // Image preview
            const imageDiv = document.createElement('div');
            imageDiv.className = 'image-preview';
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + imageResult.base64;
            img.onclick = () => openModal(img.src);
            imageDiv.appendChild(img);
            container.appendChild(imageDiv);
            
            // Analysis container
            const analysisDiv = document.createElement('div');
            analysisDiv.className = 'analysis-container';
            
            // Add each model's analysis
            imageResult.analyses.forEach(analysis => {
                const modelDiv = createModelAnalysis(analysis, imageResult.image_id);
                analysisDiv.appendChild(modelDiv);
            });
            
            container.appendChild(analysisDiv);
            section.appendChild(container);
            
            return section;
        }
        
        function createModelAnalysis(analysis, imageId) {
            const modelClass = {
                'Qwen2.5-VL-7B': 'qwen',
                'LLaVA-1.6-Mistral-7B': 'llava',
                'Pixtral-12B': 'pixtral'
            };
            
            const div = document.createElement('div');
            div.className = 'model-analysis ' + (modelClass[analysis.model] || '');
            div.dataset.model = analysis.model;
            div.dataset.confidence = analysis.confidence || 0;
            
            const confidence = (analysis.confidence || 0) * 100;
            const confClass = confidence > 80 ? 'high' : (confidence > 60 ? 'medium' : 'low');
            
            let content = `
                <div class="model-header">
                    <span class="model-name">${analysis.model}</span>
                    <div class="model-metrics">
                        <span class="confidence ${confClass}">${confidence.toFixed(0)}%</span>
                        <span>⏱️ ${(analysis.inference_time || 0).toFixed(1)}s</span>
                    </div>
                </div>
            `;
            
            if (analysis.success) {
                content += `<div class="description">${analysis.description || 'No description'}</div>`;
                
                if (analysis.ocr_text) {
                    content += `
                        <div class="ocr-section">
                            <strong>🔤 OCR Text:</strong><br>
                            ${analysis.ocr_text}
                        </div>
                    `;
                }
                
                if (analysis.extracted_data) {
                    content += `
                        <div class="data-section">
                            <strong>📊 Extracted Data:</strong><br>
                            ${JSON.stringify(analysis.extracted_data, null, 2)}
                        </div>
                    `;
                }
                
                // Evaluation controls
                const evalId = `${imageId}_${analysis.model}`;
                content += `
                    <div class="evaluation-controls">
                        <div class="rating-stars" id="stars_${evalId}">
                            <span class="star" onclick="setRating('${evalId}', 1)">★</span>
                            <span class="star" onclick="setRating('${evalId}', 2)">★</span>
                            <span class="star" onclick="setRating('${evalId}', 3)">★</span>
                            <span class="star" onclick="setRating('${evalId}', 4)">★</span>
                            <span class="star" onclick="setRating('${evalId}', 5)">★</span>
                        </div>
                        <input type="text" class="notes-input" id="notes_${evalId}" 
                               placeholder="Add notes..." 
                               onchange="updateNotes('${evalId}', this.value)">
                    </div>
                `;
            } else {
                content += `<div style="color: #dc3545;">❌ ${analysis.error_message || 'Analysis failed'}</div>`;
            }
            
            div.innerHTML = content;
            return div;
        }
        
        function setRating(evalId, rating) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].rating = rating;
            
            // Update stars display
            const stars = document.querySelectorAll(`#stars_${evalId} .star`);
            stars.forEach((star, index) => {
                star.classList.toggle('active', index < rating);
            });
            
            updateStats();
        }
        
        function updateNotes(evalId, notes) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].notes = notes;
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
        
        function applyFilters() {
            const pageFrom = parseInt(document.getElementById('pageFrom').value) || 1;
            const pageTo = parseInt(document.getElementById('pageTo').value) || 999;
            const imageType = document.getElementById('imageType').value;
            const modelFilter = document.getElementById('modelFilter').value;
            const minConfidence = parseInt(document.getElementById('confidenceFilter').value) / 100;
            
            const filtered = allResults.filter(result => {
                // Page filter
                if (result.page < pageFrom || result.page > pageTo) return false;
                
                // Type filter
                if (imageType !== 'all' && result.type !== imageType) return false;
                
                // Model/confidence filter
                if (modelFilter !== 'all' || minConfidence > 0) {
                    const hasMatchingAnalysis = result.analyses.some(a => {
                        const modelMatch = modelFilter === 'all' || a.model === modelFilter;
                        const confMatch = (a.confidence || 0) >= minConfidence;
                        return modelMatch && confMatch;
                    });
                    if (!hasMatchingAnalysis) return false;
                }
                
                return true;
            });
            
            displayResults(filtered);
        }
        
        function resetFilters() {
            document.getElementById('pageFrom').value = '';
            document.getElementById('pageTo').value = '';
            document.getElementById('imageType').value = 'all';
            document.getElementById('modelFilter').value = 'all';
            document.getElementById('confidenceFilter').value = 0;
            document.getElementById('confidenceValue').textContent = '0%';
            
            displayResults(allResults);
        }
        
        function updateStats() {
            document.getElementById('totalImages').textContent = allResults.length;
            
            let analyzed = 0;
            let totalTime = 0;
            
            allResults.forEach(result => {
                if (result.analyses && result.analyses.length > 0) {
                    analyzed++;
                    result.analyses.forEach(a => {
                        totalTime += a.inference_time || 0;
                    });
                }
            });
            
            document.getElementById('analyzedImages').textContent = analyzed;
            document.getElementById('evaluatedImages').textContent = Object.keys(evaluations).length;
            document.getElementById('totalTime').textContent = totalTime.toFixed(0);
            
            // Update summary
            updateSummary();
        }
        
        function updateSummary() {
            const summary = document.getElementById('summaryContent');
            
            // Calculate model statistics
            const modelStats = {};
            allResults.forEach(result => {
                result.analyses.forEach(analysis => {
                    if (!modelStats[analysis.model]) {
                        modelStats[analysis.model] = {
                            total: 0,
                            successful: 0,
                            totalConfidence: 0,
                            totalTime: 0,
                            hasOCR: 0
                        };
                    }
                    
                    const stats = modelStats[analysis.model];
                    stats.total++;
                    if (analysis.success) {
                        stats.successful++;
                        stats.totalConfidence += analysis.confidence || 0;
                        stats.totalTime += analysis.inference_time || 0;
                        if (analysis.ocr_text) stats.hasOCR++;
                    }
                });
            });
            
            let summaryHTML = '<table style="width: 100%; border-collapse: collapse;">';
            summaryHTML += '<tr><th>Model</th><th>Success Rate</th><th>Avg Confidence</th><th>Avg Time</th><th>OCR Rate</th></tr>';
            
            for (const [model, stats] of Object.entries(modelStats)) {
                const successRate = (stats.successful / stats.total * 100).toFixed(1);
                const avgConfidence = stats.successful > 0 ? (stats.totalConfidence / stats.successful * 100).toFixed(1) : 0;
                const avgTime = stats.successful > 0 ? (stats.totalTime / stats.successful).toFixed(1) : 0;
                const ocrRate = stats.successful > 0 ? (stats.hasOCR / stats.successful * 100).toFixed(1) : 0;
                
                summaryHTML += `
                    <tr>
                        <td>${model}</td>
                        <td>${successRate}%</td>
                        <td>${avgConfidence}%</td>
                        <td>${avgTime}s</td>
                        <td>${ocrRate}%</td>
                    </tr>
                `;
            }
            
            summaryHTML += '</table>';
            summary.innerHTML = summaryHTML;
        }
        
        function exportEvaluations() {
            const exportData = {
                timestamp: new Date().toISOString(),
                evaluations: evaluations,
                summary: {},
                detailed_results: []
            };
            
            // Add detailed results with evaluations
            allResults.forEach(result => {
                const imageEvals = {};
                result.analyses.forEach(analysis => {
                    const evalId = `${result.image_id}_${analysis.model}`;
                    if (evaluations[evalId]) {
                        imageEvals[analysis.model] = evaluations[evalId];
                    }
                });
                
                exportData.detailed_results.push({
                    image: result.description,
                    page: result.page,
                    type: result.type,
                    evaluations: imageEvals,
                    analyses: result.analyses
                });
            });
            
            // Create download
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `vlm_evaluations_${new Date().toISOString().slice(0,10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('Evaluations exported successfully!');
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>"""
    
    # Save report
    report_path = output_dir / "full_document_vlm_evaluation.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_path


def main():
    """Run complete document analysis"""
    
    logger.info("🚀 Starting Full Document VLM Analysis")
    logger.info("=" * 80)
    
    # Input PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Extract all images
    image_dir = log_dir / "extracted_images"
    extracted_images = extract_all_images_from_pdf(pdf_path, image_dir)
    
    if not extracted_images:
        logger.error("No images extracted!")
        return
    
    logger.info(f"\n📊 Will analyze {len(extracted_images)} images with 3 VLMs")
    logger.info(f"Total analyses: {len(extracted_images) * 3}")
    
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
    
    # Analyze all images
    all_results = []
    total_start = time.time()
    
    for idx, image_info in enumerate(extracted_images):
        logger.info(f"\n{'='*60}")
        logger.info(f"📸 Analyzing image {idx + 1}/{len(extracted_images)}: {image_info['description']}")
        logger.info(f"   Page: {image_info['page']}, Type: {image_info['type']}")
        
        # Prepare result structure
        image_result = {
            "image_id": f"img_{idx:03d}",
            "path": str(image_info['path']),
            "page": image_info['page'],
            "type": image_info['type'],
            "width": image_info['width'],
            "height": image_info['height'],
            "description": image_info['description'],
            "base64": image_to_base64(image_info['path']),
            "analyses": []
        }
        
        # Analyze with each VLM
        for model_name, client in clients.items():
            logger.info(f"\n   🤖 {model_name}...")
            
            analysis = analyze_image_with_vlm(client, image_info['path'], model_name)
            image_result['analyses'].append(analysis)
            
            if analysis['success']:
                logger.info(f"      ✅ Confidence: {analysis['confidence']:.0%}, Time: {analysis['inference_time']:.1f}s")
            else:
                logger.info(f"      ❌ Failed: {analysis['error_message']}")
        
        all_results.append(image_result)
        
        # Save intermediate results
        if (idx + 1) % 5 == 0:
            intermediate_file = log_dir / "intermediate_results.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Saved intermediate results ({idx + 1} images processed)")
    
    total_time = time.time() - total_start
    
    # Cleanup VLM clients
    logger.info("\n🧹 Cleaning up resources...")
    for client in clients.values():
        if hasattr(client, 'cleanup'):
            try:
                client.cleanup()
            except:
                pass
    
    # Save final results
    results_file = log_dir / "full_document_analysis_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate evaluation HTML
    logger.info("\n📄 Generating evaluation report...")
    report_path = generate_evaluation_html(all_results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🏁 ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"📊 Total images analyzed: {len(all_results)}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"⚡ Average per image: {total_time/len(all_results):.1f}s")
    
    # Model performance summary
    model_stats = {}
    for result in all_results:
        for analysis in result['analyses']:
            model = analysis['model']
            if model not in model_stats:
                model_stats[model] = {
                    'successful': 0,
                    'failed': 0,
                    'total_time': 0,
                    'confidences': []
                }
            
            if analysis['success']:
                model_stats[model]['successful'] += 1
                model_stats[model]['total_time'] += analysis['inference_time']
                model_stats[model]['confidences'].append(analysis['confidence'])
            else:
                model_stats[model]['failed'] += 1
    
    logger.info("\n📈 Model Performance:")
    for model, stats in model_stats.items():
        total = stats['successful'] + stats['failed']
        success_rate = stats['successful'] / total * 100 if total > 0 else 0
        avg_time = stats['total_time'] / stats['successful'] if stats['successful'] > 0 else 0
        avg_conf = sum(stats['confidences']) / len(stats['confidences']) * 100 if stats['confidences'] else 0
        
        logger.info(f"\n{model}:")
        logger.info(f"  • Success rate: {success_rate:.1f}% ({stats['successful']}/{total})")
        logger.info(f"  • Avg inference: {avg_time:.1f}s")
        logger.info(f"  • Avg confidence: {avg_conf:.1f}%")
    
    logger.info(f"\n📄 Evaluation report: {report_path}")
    logger.info(f"📁 Full results: {results_file}")
    logger.info(f"📂 Extracted images: {image_dir}")
    
    return {
        "report": str(report_path),
        "results": str(results_file),
        "images": str(image_dir),
        "total_time": total_time,
        "image_count": len(all_results)
    }


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Analysis complete!")
        print(f"📊 Open the evaluation report: {results['report']}")