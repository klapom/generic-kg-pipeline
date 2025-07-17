#!/usr/bin/env python3
"""
Create HTML report from existing VLM analysis results
"""

import json
from pathlib import Path
import base64
from datetime import datetime


def load_existing_results(json_path: Path) -> list:
    """Load existing analysis results"""
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_comparison_html(results: list, output_path: Path) -> None:
    """Generate comprehensive HTML comparison report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🎯 BMW Document - VLM Analysis Comparison</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        h1 { 
            font-size: 3em;
            font-weight: 300;
            letter-spacing: -1px;
            margin-bottom: 15px;
        }
        
        .subtitle {
            font-size: 1.3em;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        
        .summary-section {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin-bottom: 40px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .filter-controls {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
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
            padding: 10px 15px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 10px 25px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .image-section {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        
        .image-header {
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
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 500;
        }
        
        .image-content {
            display: grid;
            grid-template-columns: 450px 1fr;
            gap: 30px;
            align-items: start;
        }
        
        @media (max-width: 1200px) {
            .image-content {
                grid-template-columns: 1fr;
            }
        }
        
        .image-preview {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .image-preview img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            cursor: zoom-in;
            transition: transform 0.3s;
            display: block;
        }
        
        .image-preview:hover img {
            transform: scale(1.02);
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
            color: #6c757d;
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
            max-height: 200px;
            overflow-y: auto;
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
            padding: 40px;
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
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
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
        
        .no-results {
            text-align: center;
            padding: 60px;
            color: #6c757d;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 BMW Document VLM Analysis</h1>
        <div class="subtitle">Comprehensive Visual Language Model Comparison</div>
    </div>
    
    <div class="container">
"""
    
    # Calculate statistics
    total_images = len(results)
    total_analyses = sum(len(r.get('analyses', [])) for r in results)
    successful_analyses = 0
    total_confidence = 0
    total_time = 0
    
    for result in results:
        for analysis in result.get('analyses', []):
            if analysis.get('success'):
                successful_analyses += 1
                total_confidence += analysis.get('confidence', 0)
                total_time += analysis.get('inference_time', 0)
    
    avg_confidence = (total_confidence / successful_analyses * 100) if successful_analyses > 0 else 0
    
    # Summary section
    html_content += f"""
        <div class="summary-section">
            <h2>📊 Analysis Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_images}</div>
                    <div class="stat-label">Images Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_analyses}</div>
                    <div class="stat-label">Total Analyses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_confidence:.0f}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_time:.0f}s</div>
                    <div class="stat-label">Total Time</div>
                </div>
            </div>
        </div>
        
        <div class="filter-controls">
            <div class="filter-group">
                <label>Page Range:</label>
                <input type="number" id="pageFrom" placeholder="From" min="1" style="width: 80px">
                <input type="number" id="pageTo" placeholder="To" min="1" style="width: 80px">
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
            <button onclick="applyFilters()">Apply Filters</button>
            <button onclick="resetFilters()">Reset</button>
        </div>
        
        <div id="imageResults">
"""
    
    # Add each image result
    for idx, result in enumerate(results):
        image_id = result.get('image_id', f'img_{idx:03d}')
        page = result.get('page', idx + 1)
        description = result.get('description', f'Image {idx + 1}')
        
        html_content += f"""
            <div class="image-section" data-page="{page}">
                <div class="image-header">
                    <h3>{description}</h3>
                    <span class="page-badge">Page {page}</span>
                </div>
                
                <div class="image-content">
"""
        
        # Image preview (check if we have base64 data)
        if 'base64' in result:
            html_content += f"""
                    <div class="image-preview" onclick="openModal('{result['base64']}')">
                        <img src="data:image/png;base64,{result['base64']}" alt="{description}">
                    </div>
"""
        else:
            # Try to load image from path
            if 'path' in result and Path(result['path']).exists():
                try:
                    with open(result['path'], 'rb') as f:
                        base64_data = base64.b64encode(f.read()).decode('utf-8')
                    html_content += f"""
                    <div class="image-preview" onclick="openModal('{base64_data}')">
                        <img src="data:image/png;base64,{base64_data}" alt="{description}">
                    </div>
"""
                except:
                    html_content += """
                    <div class="image-preview" style="background: #f0f0f0; display: flex; align-items: center; justify-content: center; height: 300px;">
                        <span style="color: #999;">Image not available</span>
                    </div>
"""
            else:
                html_content += """
                    <div class="image-preview" style="background: #f0f0f0; display: flex; align-items: center; justify-content: center; height: 300px;">
                        <span style="color: #999;">Image not available</span>
                    </div>
"""
        
        # Analyses
        html_content += """
                    <div class="analyses-container">
"""
        
        model_classes = {
            'Qwen2.5-VL-7B': 'qwen',
            'LLaVA-1.6-Mistral-7B': 'llava',
            'Pixtral-12B': 'pixtral'
        }
        
        for analysis in result.get('analyses', []):
            model_name = analysis.get('model', 'Unknown')
            model_class = model_classes.get(model_name, '')
            
            if analysis.get('success'):
                confidence = analysis.get('confidence', 0) * 100
                conf_class = 'high' if confidence > 80 else ('medium' if confidence > 60 else 'low')
                
                html_content += f"""
                        <div class="model-analysis {model_class}" data-model="{model_name}">
                            <div class="model-header">
                                <div class="model-name">{model_name}</div>
                                <div class="metrics">
                                    <span class="confidence-badge confidence-{conf_class}">
                                        {confidence:.0f}% confidence
                                    </span>
                                    <span class="time-badge">⏱️ {analysis.get('inference_time', 0):.1f}s</span>
                                </div>
                            </div>
                            
                            <div class="description">{analysis.get('description', 'No description available')}</div>
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
                
                # Evaluation section
                eval_id = f"{image_id}_{model_name.replace('.', '_').replace('-', '_')}"
                html_content += f"""
                            <div class="evaluation-section">
                                <div class="rating-container">
                                    <span>Rate this analysis:</span>
                                    <div class="stars" id="stars_{eval_id}">
                                        {"".join([f'<span class="star" onclick="setRating(\'{eval_id}\', {i+1})">★</span>' for i in range(5)])}
                                    </div>
                                    <input type="text" 
                                           class="notes-input" 
                                           id="notes_{eval_id}"
                                           placeholder="Add notes..."
                                           onchange="updateNotes('{eval_id}', this.value)">
                                </div>
                            </div>
                        </div>
"""
            else:
                html_content += f"""
                        <div class="model-analysis {model_class}" data-model="{model_name}">
                            <div class="model-header">
                                <div class="model-name">{model_name}</div>
                                <div class="metrics">
                                    <span class="confidence-badge" style="background: #f8d7da; color: #721c24;">
                                        Failed
                                    </span>
                                </div>
                            </div>
                            <div style="color: #dc3545; font-style: italic;">
                                ❌ {analysis.get('error_message', 'Analysis failed')}
                            </div>
                        </div>
"""
        
        html_content += """
                    </div>
                </div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="export-section">
            <h2>📊 Export Your Evaluation</h2>
            <p style="margin: 20px 0; color: #6c757d;">Download all ratings, notes, and analysis results</p>
            <button class="export-btn" onclick="exportEvaluations()">
                Export All Results
            </button>
        </div>
    </div>
    
    <!-- Modal for full-size images -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        // Store all results and evaluations
        const allResults = """ + json.dumps(results) + """;
        const evaluations = {};
        let filteredResults = allResults;
        
        function applyFilters() {
            const pageFrom = parseInt(document.getElementById('pageFrom').value) || 1;
            const pageTo = parseInt(document.getElementById('pageTo').value) || 999;
            const modelFilter = document.getElementById('modelFilter').value;
            
            // Hide all sections first
            document.querySelectorAll('.image-section').forEach(section => {
                section.style.display = 'none';
            });
            
            // Show filtered sections
            document.querySelectorAll('.image-section').forEach(section => {
                const page = parseInt(section.dataset.page);
                if (page >= pageFrom && page <= pageTo) {
                    if (modelFilter === 'all') {
                        section.style.display = 'block';
                    } else {
                        // Check if section has the selected model
                        const hasModel = section.querySelector(`.model-analysis[data-model="${modelFilter}"]`);
                        if (hasModel) {
                            section.style.display = 'block';
                            // Hide other models
                            section.querySelectorAll('.model-analysis').forEach(analysis => {
                                analysis.style.display = analysis.dataset.model === modelFilter ? 'block' : 'none';
                            });
                        }
                    }
                }
            });
        }
        
        function resetFilters() {
            document.getElementById('pageFrom').value = '';
            document.getElementById('pageTo').value = '';
            document.getElementById('modelFilter').value = 'all';
            
            // Show all
            document.querySelectorAll('.image-section').forEach(section => {
                section.style.display = 'block';
            });
            document.querySelectorAll('.model-analysis').forEach(analysis => {
                analysis.style.display = 'block';
            });
        }
        
        function setRating(evalId, rating) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].rating = rating;
            
            // Update stars display
            const stars = document.querySelectorAll(`#stars_${evalId} .star`);
            stars.forEach((star, index) => {
                star.classList.toggle('active', index < rating);
            });
        }
        
        function updateNotes(evalId, notes) {
            evaluations[evalId] = evaluations[evalId] || {};
            evaluations[evalId].notes = notes;
        }
        
        function openModal(base64) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            modal.style.display = 'block';
            modalImg.src = 'data:image/png;base64,' + base64;
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        function exportEvaluations() {
            const exportData = {
                timestamp: new Date().toISOString(),
                document: 'BMW Document Analysis',
                total_images: allResults.length,
                evaluations: evaluations,
                summary: calculateSummary(),
                detailed_results: allResults.map(result => ({
                    ...result,
                    evaluations: Object.keys(evaluations)
                        .filter(key => key.startsWith(result.image_id || 'img'))
                        .reduce((acc, key) => {
                            acc[key] = evaluations[key];
                            return acc;
                        }, {})
                }))
            };
            
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
        
        function calculateSummary() {
            const modelStats = {};
            
            allResults.forEach(result => {
                result.analyses?.forEach(analysis => {
                    const model = analysis.model;
                    if (!modelStats[model]) {
                        modelStats[model] = {
                            total: 0,
                            successful: 0,
                            totalConfidence: 0,
                            totalTime: 0
                        };
                    }
                    
                    modelStats[model].total++;
                    if (analysis.success) {
                        modelStats[model].successful++;
                        modelStats[model].totalConfidence += analysis.confidence || 0;
                        modelStats[model].totalTime += analysis.inference_time || 0;
                    }
                });
            });
            
            return modelStats;
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
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated: {output_path}")


def main():
    # Check for existing results
    results_path = Path("tests/debugging/full_document_vlm_analysis/intermediate_results.json")
    
    if results_path.exists():
        print(f"📄 Loading results from: {results_path}")
        results = load_existing_results(results_path)
        print(f"✅ Loaded {len(results)} image analyses")
        
        # Generate HTML
        output_path = Path("tests/debugging/full_document_vlm_analysis/bmw_vlm_comparison_report.html")
        generate_comparison_html(results, output_path)
        
        print(f"\n🌐 Open the report in your browser:")
        print(f"   file://{output_path.absolute()}")
    else:
        print("❌ No results file found!")


if __name__ == "__main__":
    main()