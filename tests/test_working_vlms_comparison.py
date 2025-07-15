#!/usr/bin/env python3
"""
Working VLMs Comparison Test

Tests Qwen2.5-VL (Transformers) and Llama-3.2-Vision (vLLM) for comparison.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/working_vlms_comparison")
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
        },
        {
            "path": test_images_dir / "visual_page5_element1.png",
            "name": "BMW interior seats",
            "page": 5,
            "element_type": VisualElementType.IMAGE,
            "expected_features": ["seats", "leather", "interior design"]
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

def generate_comparison_html(results: List[Dict[str, Any]], output_dir: Path):
    """Generate HTML comparison report for working VLMs"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>VLM Comparison Report - Qwen2.5-VL vs Llama-3.2-Vision</title>
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
        .comparison-container {
            max-width: 1400px;
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
        .image-info h2 {
            margin: 0 0 10px 0;
            color: #0066cc;
        }
        .models-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
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
            font-size: 1.2em;
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
            max-height: 200px;
            overflow-y: auto;
        }
        .processing-time {
            color: #666;
            font-size: 0.85em;
            margin-top: 10px;
        }
        .consensus {
            margin-top: 20px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 5px;
            border: 1px solid #2196f3;
            text-align: center;
        }
        .comparison-metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #0066cc;
        }
        .metric-label {
            color: #666;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
    <h1>🔍 VLM Comparison: Qwen2.5-VL vs Llama-3.2-Vision</h1>
"""
    
    # Calculate summary statistics
    total_images = len(results)
    qwen_wins = sum(1 for r in results if r.get("consensus", {}).get("best_model") == "qwen2.5-vl-7b")
    llama_wins = sum(1 for r in results if r.get("consensus", {}).get("best_model") == "llama-3.2-11b-vision")
    
    # Average processing times
    qwen_times = []
    llama_times = []
    for result in results:
        models = result.get("model_comparisons", {})
        if "qwen2.5-vl-7b" in models:
            qwen_times.append(models["qwen2.5-vl-7b"].get("processing_time_seconds", 0))
        if "llama-3.2-11b-vision" in models:
            llama_times.append(models["llama-3.2-11b-vision"].get("processing_time_seconds", 0))
    
    avg_qwen_time = sum(qwen_times) / len(qwen_times) if qwen_times else 0
    avg_llama_time = sum(llama_times) / len(llama_times) if llama_times else 0
    
    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="comparison-metrics">
            <div class="metric">
                <div class="metric-value">{total_images}</div>
                <div class="metric-label">Images Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{qwen_wins}</div>
                <div class="metric-label">Qwen2.5-VL Wins</div>
            </div>
            <div class="metric">
                <div class="metric-value">{llama_wins}</div>
                <div class="metric-label">Llama-3.2 Wins</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_images - qwen_wins - llama_wins}</div>
                <div class="metric-label">Ties</div>
            </div>
        </div>
        <div class="comparison-metrics" style="margin-top: 15px;">
            <div class="metric">
                <div class="metric-value">{avg_qwen_time:.2f}s</div>
                <div class="metric-label">Avg Qwen2.5-VL Time</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_llama_time:.2f}s</div>
                <div class="metric-label">Avg Llama-3.2 Time</div>
            </div>
            <div class="metric">
                <div class="metric-value">{'Qwen' if avg_qwen_time < avg_llama_time else 'Llama'}</div>
                <div class="metric-label">Faster Model</div>
            </div>
            <div class="metric">
                <div class="metric-value">{abs(avg_qwen_time - avg_llama_time):.2f}s</div>
                <div class="metric-label">Time Difference</div>
            </div>
        </div>
    </div>
    
    <div class="comparison-container">
"""
    
    # Add individual comparisons
    for i, result in enumerate(results):
        image_name = result.get("image_name", "Unknown")
        page = result.get("page", 0)
        consensus = result.get("consensus", {})
        best_model = consensus.get("best_model")
        models = result.get("model_comparisons", {})
        
        # Get image path
        image_path = None
        for f in (log_dir.parent / "visual_elements_with_vlm" / "extracted_images").glob("*.png"):
            if f"page{page}" in f.name and f"element{i}" in f.name:
                image_path = f
                break
        
        html_content += f"""
        <div class="image-comparison">
            <div class="image-header">
"""
        
        if image_path and image_path.exists():
            # Use relative path for HTML
            rel_path = f"../visual_elements_with_vlm/extracted_images/{image_path.name}"
            html_content += f'<img src="{rel_path}" class="image-thumbnail" alt="{image_name}">'
        
        html_content += f"""
                <div class="image-info">
                    <h2>{image_name}</h2>
                    <p><strong>Page:</strong> {page}</p>
                    <p><strong>Best Model:</strong> <span style="color: #4caf50; font-weight: bold;">{best_model or 'Tie'}</span></p>
                    <p><strong>Consensus Score:</strong> {consensus.get('score', 0):.2f}</p>
                </div>
            </div>
            
            <div class="models-comparison">
"""
        
        # Add Qwen2.5-VL results
        if "qwen2.5-vl-7b" in models:
            qwen_data = models["qwen2.5-vl-7b"]
            is_best = best_model == "qwen2.5-vl-7b"
            confidence = qwen_data.get("confidence", 0)
            confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
            
            html_content += f"""
                <div class="model-result {'best' if is_best else ''}">
                    <div class="model-header">
                        <span>Qwen2.5-VL-7B (Transformers) {'⭐' if is_best else ''}</span>
                        <span class="confidence {confidence_class}">
                            {confidence:.1%}
                        </span>
                    </div>
                    <div class="processing-time">
                        Processing: {qwen_data.get('processing_time_seconds', 0):.2f}s
                    </div>
                    <div class="description">
                        <strong>Description:</strong><br>
                        {qwen_data.get('description', 'No description available')}
                    </div>
"""
            
            if qwen_data.get('extracted_text'):
                html_content += f"""
                    <div class="ocr-text">
                        <strong>OCR Text:</strong><br>
                        {qwen_data.get('extracted_text', '')}
                    </div>
"""
            
            html_content += "</div>"
        
        # Add Llama results
        if "llama-3.2-11b-vision" in models:
            llama_data = models["llama-3.2-11b-vision"]
            is_best = best_model == "llama-3.2-11b-vision"
            confidence = llama_data.get("confidence", 0)
            confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
            
            html_content += f"""
                <div class="model-result {'best' if is_best else ''}">
                    <div class="model-header">
                        <span>Llama-3.2-11B-Vision (vLLM) {'⭐' if is_best else ''}</span>
                        <span class="confidence {confidence_class}">
                            {confidence:.1%}
                        </span>
                    </div>
                    <div class="processing-time">
                        Processing: {llama_data.get('processing_time_seconds', 0):.2f}s
                    </div>
                    <div class="description">
                        <strong>Description:</strong><br>
                        {llama_data.get('description', 'No description available')}
                    </div>
"""
            
            if llama_data.get('extracted_text'):
                html_content += f"""
                    <div class="ocr-text">
                        <strong>OCR Text:</strong><br>
                        {llama_data.get('extracted_text', '')}
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
    report_path = output_dir / "working_vlms_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {report_path}")
    return report_path

async def main():
    """Run VLM comparison with working models"""
    
    # Only use models that work on our GPU
    test_models = [
        VLMModelType.QWEN25_VL_7B,      # Works with Transformers fallback
        VLMModelType.LLAMA32_VISION_11B, # Works with vLLM
    ]
    
    logger.info(f"Initializing VLM comparison with {len(test_models)} models...")
    
    # Initialize multi-VLM integration
    multi_vlm = MultiVLMIntegration(enabled_models=test_models)
    
    # Load test images
    visual_elements = load_test_images()
    logger.info(f"Loaded {len(visual_elements)} test images")
    
    # Run comparative analysis
    results_dict = await multi_vlm.analyze_visual_elements_comparative(visual_elements)
    
    # Save JSON results
    json_results = []
    for i, element in enumerate(visual_elements):
        if element.content_hash not in results_dict:
            continue
        comparison = results_dict[element.content_hash]
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
    
    results_file = log_dir / "working_vlms_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(json_results, log_dir)
    
    # Print detailed comparison
    logger.info("\n" + "="*60)
    logger.info("DETAILED COMPARISON RESULTS")
    logger.info("="*60)
    
    for result in json_results:
        logger.info(f"\n📸 Image: {result['image_name']}")
        logger.info(f"Best Model: {result['consensus']['best_model'] or 'Tie'}")
        logger.info(f"Consensus Score: {result['consensus']['score']:.2f}")
        
        models = result['model_comparisons']
        if "qwen2.5-vl-7b" in models and "llama-3.2-11b-vision" in models:
            qwen = models["qwen2.5-vl-7b"]
            llama = models["llama-3.2-11b-vision"]
            
            logger.info(f"\n  Qwen2.5-VL-7B:")
            logger.info(f"    - Confidence: {qwen['confidence']:.1%}")
            logger.info(f"    - Processing: {qwen['processing_time_seconds']:.2f}s")
            logger.info(f"    - Description: {qwen['description'][:100]}...")
            
            logger.info(f"\n  Llama-3.2-11B-Vision:")
            logger.info(f"    - Confidence: {llama['confidence']:.1%}")
            logger.info(f"    - Processing: {llama['processing_time_seconds']:.2f}s")
            logger.info(f"    - Description: {llama['description'][:100]}...")
    
    logger.info("\n✅ Comparison complete!")
    logger.info(f"📊 View report at: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())