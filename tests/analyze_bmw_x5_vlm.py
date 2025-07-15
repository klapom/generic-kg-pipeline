#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
import json
import base64
import time
import logging
from datetime import datetime
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType
from core.parsers.hybrid_pdf_parser import HybridPDFParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def analyze_bmw_x5_document():
    """Analyze BMW X5 G05 document with all VLMs"""
    
    # Setup paths
    pdf_path = Path("data/input/Preview_BMW_X5_G05.pdf")
    output_dir = Path("tests/debugging/bmw_x5_vlm_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract images from PDF
    logger.info(f"📄 Extracting images from {pdf_path.name}...")
    
    parser = HybridPDFParser()
    result = await parser.parse(pdf_path)
    
    # Collect all visuals from all pages
    all_visuals = []
    for page_idx, page in enumerate(result.pages):
        logger.info(f"Page {page_idx + 1}: {len(page.visuals)} visuals found")
        for visual in page.visuals:
            # Save visual
            visual_path = output_dir / f"page_{page_idx+1:03d}_{visual.element_type.value}_{visual.element_id}.png"
            with open(visual_path, 'wb') as f:
                f.write(visual.content)
            
            all_visuals.append({
                "page": page_idx + 1,
                "element_id": visual.element_id,
                "element_type": visual.element_type.value,
                "path": str(visual_path),
                "content": base64.b64encode(visual.content).decode('utf-8')
            })
    
    logger.info(f"\\n📸 Extracted {len(all_visuals)} visuals total")
    
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
    
    # Analyze each visual with all VLMs
    results = []
    
    for visual_idx, visual in enumerate(all_visuals):
        logger.info(f"\\n{'='*80}")
        logger.info(f"🖼️ Analyzing visual {visual_idx + 1}/{len(all_visuals)}: Page {visual['page']}, {visual['element_type']}")
        logger.info(f"{'='*80}")
        
        visual_result = {
            "image_id": f"page_{visual['page']:03d}_{visual['element_type']}_{visual['element_id']}",
            "page": visual['page'],
            "element_type": visual['element_type'],
            "path": visual['path'],
            "base64": visual['content'],
            "analyses": []
        }
        
        # Test each model
        for model_name, client in clients.items():
            logger.info(f"\\n🤖 Testing {model_name}...")
            
            try:
                # Decode image data
                image_data = base64.b64decode(visual['content'])
                
                start = time.time()
                result = client.analyze_visual(
                    image_data=image_data,
                    element_type=VisualElementType(visual['element_type']) if visual['element_type'] != 'full_page' else None,
                    analysis_focus="comprehensive"
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
            
            # Cleanup after each model to save memory
            if hasattr(client, "cleanup"):
                try:
                    client.cleanup()
                except:
                    pass
        
        results.append(visual_result)
        
        # Save intermediate results
        intermediate_file = output_dir / "intermediate_results.json"
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"bmw_x5_vlm_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\\n✅ Results saved to: {results_file}")
    
    # Generate HTML report
    generate_html_report(results, output_dir / f"bmw_x5_vlm_comparison_{timestamp}.html")
    
    return results

def generate_html_report(results, output_path):
    """Generate comprehensive HTML report with filtering and comparison features"""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW X5 G05 VLM Comparison Report</title>
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
        .filters {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .filter-group {{
            display: inline-block;
            margin-right: 20px;
        }}
        .image-comparison {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }}
        .image-header {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .image-preview {{
            width: 200px;
            margin-right: 20px;
        }}
        .image-preview img {{
            width: 100%;
            border: 1px solid #ddd;
            cursor: pointer;
        }}
        .image-info {{
            flex-grow: 1;
        }}
        .analyses-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .analysis {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #f9f9f9;
        }}
        .analysis-header {{
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ddd;
        }}
        .analysis-content {{
            margin-top: 10px;
        }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .confidence {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            font-size: 0.9em;
        }}
        .inference-time {{
            color: #666;
            font-size: 0.9em;
        }}
        .description {{
            margin-top: 10px;
            padding: 10px;
            background-color: white;
            border-left: 3px solid #007bff;
            max-height: 200px;
            overflow-y: auto;
        }}
        .ocr-text {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f0f8ff;
            border-left: 3px solid #17a2b8;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
        }}
        .extracted-data {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f0fff0;
            border-left: 3px solid #28a745;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 150px;
            overflow-y: auto;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            margin: 5% auto;
            max-width: 90%;
            max-height: 80%;
        }}
        .modal-content img {{
            width: 100%;
            height: auto;
        }}
        .close {{
            position: absolute;
            right: 20px;
            top: 20px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }}
        .stats {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .rating {{
            margin-top: 10px;
        }}
        .rating button {{
            margin-right: 5px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            background-color: white;
            cursor: pointer;
            border-radius: 4px;
        }}
        .rating button:hover {{
            background-color: #f0f0f0;
        }}
        .rating button.selected {{
            background-color: #007bff;
            color: white;
        }}
        .export-section {{
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }}
        .highlight-diagram {{
            background-color: #fff3cd;
            border: 2px solid #ffc107;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BMW X5 G05 - VLM Comparison Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Total Images: {total_images} | Models: Qwen2.5-VL, LLaVA-1.6, Pixtral-12B</p>
    </div>

    <div class="stats">
        <h2>Summary Statistics</h2>
        <div id="stats-content"></div>
    </div>

    <div class="filters">
        <h3>Filters</h3>
        <div class="filter-group">
            <label>Page: 
                <select id="pageFilter">
                    <option value="">All Pages</option>
                </select>
            </label>
        </div>
        <div class="filter-group">
            <label>Element Type: 
                <select id="typeFilter">
                    <option value="">All Types</option>
                </select>
            </label>
        </div>
        <div class="filter-group">
            <label>Model: 
                <select id="modelFilter">
                    <option value="">All Models</option>
                    <option value="Qwen2.5-VL-7B">Qwen2.5-VL-7B</option>
                    <option value="LLaVA-1.6-Mistral-7B">LLaVA-1.6-Mistral-7B</option>
                    <option value="Pixtral-12B">Pixtral-12B</option>
                </select>
            </label>
        </div>
        <div class="filter-group">
            <label>
                <input type="checkbox" id="diagramsOnly"> Show Diagrams Only
            </label>
        </div>
    </div>

    <div id="comparisons-container"></div>

    <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <div class="modal-content">
            <img id="modalImage" src="">
        </div>
    </div>

    <div class="export-section">
        <h3>Export Results</h3>
        <button onclick="exportRatings()">Export Ratings as CSV</button>
        <button onclick="exportAnalyses()">Export All Analyses as JSON</button>
    </div>

    <script>
        const results = {results_json};
        let ratings = {{}};

        function initializeFilters() {{
            const pages = [...new Set(results.map(r => r.page))].sort((a, b) => a - b);
            const types = [...new Set(results.map(r => r.element_type))].sort();
            
            const pageFilter = document.getElementById('pageFilter');
            pages.forEach(page => {{
                const option = document.createElement('option');
                option.value = page;
                option.textContent = `Page ${{page}}`;
                pageFilter.appendChild(option);
            }});
            
            const typeFilter = document.getElementById('typeFilter');
            types.forEach(type => {{
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type;
                typeFilter.appendChild(option);
            }});
        }}

        function updateStats() {{
            const stats = {{
                totalImages: results.length,
                byPage: {{}},
                byType: {{}},
                modelPerformance: {{}}
            }};
            
            results.forEach(result => {{
                // Count by page
                stats.byPage[`Page ${{result.page}}`] = (stats.byPage[`Page ${{result.page}}`] || 0) + 1;
                
                // Count by type
                stats.byType[result.element_type] = (stats.byType[result.element_type] || 0) + 1;
                
                // Model performance
                result.analyses.forEach(analysis => {{
                    if (!stats.modelPerformance[analysis.model]) {{
                        stats.modelPerformance[analysis.model] = {{
                            success: 0,
                            total: 0,
                            avgConfidence: 0,
                            avgTime: 0
                        }};
                    }}
                    const perf = stats.modelPerformance[analysis.model];
                    perf.total++;
                    if (analysis.success) {{
                        perf.success++;
                        perf.avgConfidence += analysis.confidence;
                    }}
                    perf.avgTime += analysis.inference_time;
                }});
            }});
            
            // Calculate averages
            Object.keys(stats.modelPerformance).forEach(model => {{
                const perf = stats.modelPerformance[model];
                perf.avgConfidence = perf.success > 0 ? (perf.avgConfidence / perf.success * 100).toFixed(1) : 0;
                perf.avgTime = (perf.avgTime / perf.total).toFixed(1);
                perf.successRate = ((perf.success / perf.total) * 100).toFixed(1);
            }});
            
            // Display stats
            let statsHtml = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">';
            
            // Pages distribution
            statsHtml += '<div><h4>Pages Distribution</h4><ul>';
            Object.entries(stats.byPage).forEach(([page, count]) => {{
                statsHtml += `<li>${{page}}: ${{count}} images</li>`;
            }});
            statsHtml += '</ul></div>';
            
            // Types distribution
            statsHtml += '<div><h4>Element Types</h4><ul>';
            Object.entries(stats.byType).forEach(([type, count]) => {{
                statsHtml += `<li>${{type}}: ${{count}}</li>`;
            }});
            statsHtml += '</ul></div>';
            
            // Model performance
            statsHtml += '<div><h4>Model Performance</h4><ul>';
            Object.entries(stats.modelPerformance).forEach(([model, perf]) => {{
                statsHtml += `<li><strong>${{model}}</strong><br>
                    Success Rate: ${{perf.successRate}}%<br>
                    Avg Confidence: ${{perf.avgConfidence}}%<br>
                    Avg Time: ${{perf.avgTime}}s</li>`;
            }});
            statsHtml += '</ul></div>';
            
            statsHtml += '</div>';
            document.getElementById('stats-content').innerHTML = statsHtml;
        }}

        function renderComparisons() {{
            const container = document.getElementById('comparisons-container');
            const pageFilter = document.getElementById('pageFilter').value;
            const typeFilter = document.getElementById('typeFilter').value;
            const modelFilter = document.getElementById('modelFilter').value;
            const diagramsOnly = document.getElementById('diagramsOnly').checked;
            
            let filteredResults = results;
            
            if (pageFilter) {{
                filteredResults = filteredResults.filter(r => r.page == pageFilter);
            }}
            if (typeFilter) {{
                filteredResults = filteredResults.filter(r => r.element_type === typeFilter);
            }}
            if (diagramsOnly) {{
                filteredResults = filteredResults.filter(r => r.element_type === 'diagram');
            }}
            
            container.innerHTML = '';
            
            filteredResults.forEach(result => {{
                const compDiv = document.createElement('div');
                compDiv.className = 'image-comparison';
                if (result.element_type === 'diagram') {{
                    compDiv.classList.add('highlight-diagram');
                }}
                
                // Image header
                const headerHtml = `
                    <div class="image-header">
                        <div class="image-preview">
                            <img src="data:image/png;base64,${{result.base64}}" 
                                 onclick="showModal(this.src)" 
                                 alt="${{result.image_id}}">
                        </div>
                        <div class="image-info">
                            <h3>${{result.image_id}}</h3>
                            <p><strong>Page:</strong> ${{result.page}} | 
                               <strong>Type:</strong> ${{result.element_type}}
                               ${{result.element_type === 'diagram' ? ' ⚠️ <strong>DIAGRAM</strong>' : ''}}
                            </p>
                        </div>
                    </div>
                `;
                
                // Analyses
                let analysesHtml = '<div class="analyses-container">';
                
                let filteredAnalyses = result.analyses;
                if (modelFilter) {{
                    filteredAnalyses = filteredAnalyses.filter(a => a.model === modelFilter);
                }}
                
                filteredAnalyses.forEach(analysis => {{
                    const ratingKey = `${{result.image_id}}_${{analysis.model}}`;
                    analysesHtml += `
                        <div class="analysis">
                            <div class="analysis-header">
                                <span>${{analysis.model}}</span>
                                <span class="${{analysis.success ? 'success' : 'failure'}}">
                                    ${{analysis.success ? '✅ Success' : '❌ Failed'}}
                                </span>
                                ${{analysis.success ? `<span class="confidence">${{(analysis.confidence * 100).toFixed(0)}}%</span>` : ''}}
                                <span class="inference-time">${{analysis.inference_time.toFixed(1)}}s</span>
                            </div>
                            <div class="analysis-content">
                                ${{analysis.success ? `
                                    <div class="description">
                                        <strong>Description:</strong><br>
                                        ${{analysis.description}}
                                    </div>
                                    ${{analysis.ocr_text ? `
                                        <div class="ocr-text">
                                            <strong>OCR Text:</strong><br>
                                            ${{analysis.ocr_text}}
                                        </div>
                                    ` : ''}}
                                    ${{analysis.extracted_data ? `
                                        <div class="extracted-data">
                                            <strong>Extracted Data:</strong><br>
                                            ${{JSON.stringify(analysis.extracted_data, null, 2)}}
                                        </div>
                                    ` : ''}}
                                    <div class="rating">
                                        <strong>Rate this analysis:</strong><br>
                                        <button onclick="setRating('${{ratingKey}}', 1)" class="${{ratings[ratingKey] === 1 ? 'selected' : ''}}">👎 Poor</button>
                                        <button onclick="setRating('${{ratingKey}}', 2)" class="${{ratings[ratingKey] === 2 ? 'selected' : ''}}">😐 Fair</button>
                                        <button onclick="setRating('${{ratingKey}}', 3)" class="${{ratings[ratingKey] === 3 ? 'selected' : ''}}">👍 Good</button>
                                        <button onclick="setRating('${{ratingKey}}', 4)" class="${{ratings[ratingKey] === 4 ? 'selected' : ''}}">🌟 Excellent</button>
                                    </div>
                                ` : `
                                    <div class="failure">
                                        Error: ${{analysis.error_message}}
                                    </div>
                                `}}
                            </div>
                        </div>
                    `;
                }});
                
                analysesHtml += '</div>';
                
                compDiv.innerHTML = headerHtml + analysesHtml;
                container.appendChild(compDiv);
            }});
        }}

        function showModal(src) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }}

        function setRating(key, value) {{
            ratings[key] = value;
            renderComparisons();
        }}

        function exportRatings() {{
            let csv = 'Image ID,Model,Rating\\n';
            Object.entries(ratings).forEach(([key, rating]) => {{
                const [imageId, model] = key.split('_').slice(0, -1).join('_').split('_');
                csv += `${{imageId}},${{model}},${{rating}}\\n`;
            }});
            
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'vlm_ratings.csv';
            a.click();
        }}

        function exportAnalyses() {{
            const dataStr = JSON.stringify(results, null, 2);
            const blob = new Blob([dataStr], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'vlm_analyses.json';
            a.click();
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            initializeFilters();
            updateStats();
            renderComparisons();
            
            // Event listeners
            document.getElementById('pageFilter').addEventListener('change', renderComparisons);
            document.getElementById('typeFilter').addEventListener('change', renderComparisons);
            document.getElementById('modelFilter').addEventListener('change', renderComparisons);
            document.getElementById('diagramsOnly').addEventListener('change', renderComparisons);
            
            // Modal close
            document.querySelector('.close').addEventListener('click', () => {{
                document.getElementById('imageModal').style.display = 'none';
            }});
            
            window.addEventListener('click', (e) => {{
                const modal = document.getElementById('imageModal');
                if (e.target === modal) {{
                    modal.style.display = 'none';
                }}
            }});
        }});
    </script>
</body>
</html>'''
    
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_images=len(results),
        results_json=json.dumps(results)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report generated: {output_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(analyze_bmw_x5_document())