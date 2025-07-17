#!/usr/bin/env python3
"""
Public VLMs Comparison Test

Tests publicly available VLMs that don't require authentication:
- Qwen2.5-VL-7B (Transformers)
- LLaVA-1.6-Mistral-7B (Transformers) 
- Pixtral-12B (Transformers)
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/public_vlms_comparison")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'comparison_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, '.')

from core.parsers.vlm_integration import MultiVLMIntegration, VLMModelType
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

def load_test_images() -> List[VisualElement]:
    """Load test images for comparison"""
    test_images_dir = Path("tests/debugging/visual_elements_with_vlm/extracted_images")
    
    # Define test images with metadata
    test_cases = [
        {
            "path": test_images_dir / "visual_page3_element2.png",
            "name": "BMW front view with annotations",
            "page": 3,
            "element_type": VisualElementType.IMAGE,
            "expected_features": ["BMW grille", "annotations", "technical labels"]
        },
        {
            "path": test_images_dir / "visual_page5_element0.png", 
            "name": "BMW interior dashboard",
            "page": 5,
            "element_type": VisualElementType.IMAGE,
            "expected_features": ["dashboard", "steering wheel", "interior"]
        }
    ]
    
    visual_elements = []
    
    for test_case in test_cases:
        if not test_case["path"].exists():
            logger.warning(f"Test image not found: {test_case['path']}")
            continue
            
        logger.info(f"Loading image: {test_case['name']}")
        
        with open(test_case["path"], 'rb') as f:
            image_data = f.read()
        
        element = VisualElement(
            element_type=test_case["element_type"],
            source_format=DocumentType.PDF,
            content_hash=f"test_hash_{test_case['name']}",
            raw_data=image_data,
            page_or_slide=test_case["page"],
            confidence=1.0,
            file_extension="png"
        )
        # Add metadata
        element.analysis_metadata = {
            "name": test_case["name"],
            "expected_features": test_case["expected_features"]
        }
        
        visual_elements.append(element)
    
    return visual_elements

def generate_comparison_html(results: List[Dict[str, Any]], output_dir: Path, models_tested: List[str]):
    """Generate HTML comparison report for public VLMs"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Public VLMs Comparison Report</title>
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
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 1200px;
            margin: 0 auto 30px;
        }
        .models-list {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .comparison-container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .image-comparison {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }
        .image-thumbnail {
            width: 300px;
            height: 200px;
            object-fit: contain;
            border: 1px solid #ddd;
            margin-right: 30px;
            background: #f9f9f9;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .model-result {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            background: #f9f9f9;
        }
        .model-result.best {
            border-color: #4caf50;
            background: #e8f5e9;
        }
        .model-header {
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 15px;
            font-size: 1.1em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .confidence {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .confidence.high { background: #4caf50; color: white; }
        .confidence.medium { background: #ff9800; color: white; }
        .confidence.low { background: #f44336; color: white; }
        .description {
            margin-top: 15px;
            color: #333;
            line-height: 1.8;
            max-height: 300px;
            overflow-y: auto;
        }
        .ocr-text {
            margin-top: 15px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 150px;
            overflow-y: auto;
        }
        .processing-time {
            color: #666;
            font-size: 0.85em;
            margin-top: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #0066cc;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>🔍 Public VLMs Comparison Report</h1>
"""
    
    # Calculate statistics
    total_images = len(results)
    model_wins = {}
    model_times = {}
    
    for model in models_tested:
        model_wins[model] = sum(1 for r in results if r.get("consensus", {}).get("best_model") == model)
        times = []
        for result in results:
            if model in result.get("model_comparisons", {}):
                times.append(result["model_comparisons"][model].get("processing_time_seconds", 0))
        model_times[model] = sum(times) / len(times) if times else 0
    
    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="models-list">
            <strong>Models Tested:</strong> {', '.join(models_tested)}
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_images}</div>
                <div class="stat-label">Images Analyzed</div>
            </div>
"""
    
    for model in models_tested:
        html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{model_wins[model]}</div>
                <div class="stat-label">{model.split('-')[0]} Wins</div>
            </div>
"""
    
    html_content += """
        </div>
        
        <h3>Average Processing Times</h3>
        <div class="stats-grid">
"""
    
    for model, avg_time in model_times.items():
        html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{avg_time:.2f}s</div>
                <div class="stat-label">{model}</div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
    
    <div class="comparison-container">
"""
    
    # Add individual comparisons
    for result in results:
        image_name = result.get("image_name", "Unknown")
        page = result.get("page", 0)
        consensus = result.get("consensus", {})
        best_model = consensus.get("best_model")
        models = result.get("model_comparisons", {})
        
        html_content += f"""
        <div class="image-comparison">
            <div class="image-header">
                <div class="image-info">
                    <h2>{image_name}</h2>
                    <p><strong>Page:</strong> {page}</p>
                    <p><strong>Best Model:</strong> <span style="color: #4caf50; font-weight: bold;">{best_model or 'Tie'}</span></p>
                    <p><strong>Consensus Score:</strong> {consensus.get('score', 0):.2f}</p>
                </div>
            </div>
            
            <div class="models-grid">
"""
        
        # Add results for each model
        for model_name, model_data in models.items():
            is_best = best_model == model_name
            confidence = model_data.get("confidence", 0)
            confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
            
            html_content += f"""
                <div class="model-result {'best' if is_best else ''}">
                    <div class="model-header">
                        <span>{model_name} {'⭐' if is_best else ''}</span>
                        <span class="confidence {confidence_class}">
                            {confidence:.1%}
                        </span>
                    </div>
                    <div class="processing-time">
                        Processing: {model_data.get('processing_time_seconds', 0):.2f}s
                    </div>
                    <div class="description">
                        <strong>Description:</strong><br>
                        {model_data.get('description', 'No description available')}
                    </div>
"""
            
            if model_data.get('extracted_text'):
                html_content += f"""
                    <div class="ocr-text">
                        <strong>OCR Text:</strong><br>
                        {model_data.get('extracted_text', '')}
                    </div>
"""
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "public_vlms_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {report_path}")
    return report_path

async def main():
    """Run public VLMs comparison"""
    
    # Only use publicly available models
    test_models = [
        VLMModelType.QWEN25_VL_7B,   # Public model
        VLMModelType.LLAVA_16_7B,    # Public model (Mistral-based)
        VLMModelType.PIXTRAL_12B,    # Public model
    ]
    
    logger.info(f"Initializing comparison with {len(test_models)} public VLMs...")
    
    # Initialize multi-VLM integration
    multi_vlm = MultiVLMIntegration(enabled_models=test_models)
    
    # Load test images
    visual_elements = load_test_images()
    logger.info(f"Loaded {len(visual_elements)} test images")
    
    if not visual_elements:
        logger.error("No test images found!")
        return
    
    # Run comparative analysis
    results_dict = await multi_vlm.analyze_visual_elements_comparative(visual_elements)
    
    # Save JSON results
    json_results = []
    models_tested = []
    
    for element in visual_elements:
        if element.content_hash not in results_dict:
            continue
        comparison = results_dict[element.content_hash]
        
        # Track which models were actually tested
        for model in comparison.model_results.keys():
            if model not in models_tested:
                models_tested.append(model)
        
        result_data = {
            "image_name": element.analysis_metadata.get("name", "Unknown"),
            "page": element.page_or_slide or 0,
            "model_comparisons": comparison.model_results,
            "consensus": {
                "score": comparison.consensus_score,
                "best_model": comparison.best_model
            },
            "timestamp": comparison.timestamp
        }
        json_results.append(result_data)
    
    results_file = log_dir / "public_vlms_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(json_results, log_dir, models_tested)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PUBLIC VLMS COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Models tested: {', '.join(models_tested)}")
    
    for result in json_results:
        logger.info(f"\n📸 Image: {result['image_name']}")
        logger.info(f"Best Model: {result['consensus']['best_model'] or 'Tie'}")
        logger.info(f"Consensus Score: {result['consensus']['score']:.2f}")
        
        for model, data in result['model_comparisons'].items():
            logger.info(f"\n  {model}:")
            logger.info(f"    - Confidence: {data['confidence']:.1%}")
            logger.info(f"    - Processing: {data['processing_time_seconds']:.2f}s")
    
    logger.info("\n✅ Comparison complete!")
    logger.info(f"📊 View report at: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())