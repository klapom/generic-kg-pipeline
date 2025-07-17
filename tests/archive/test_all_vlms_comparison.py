#!/usr/bin/env python3
"""
Comprehensive Multi-VLM Comparison Test

Tests all available VLMs and generates a detailed comparison report.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/all_vlms_comparison")
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
            source_format=DocumentType.PDF,  # Assume PDF for test
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
    """Generate detailed HTML comparison report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive VLM Comparison Report</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }
        h1, h2 { 
            color: #333; 
            text-align: center;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            width: 200px;
            height: 200px;
            object-fit: contain;
            border: 1px solid #ddd;
            margin-right: 20px;
            background: #f9f9f9;
        }
        .image-info h2 {
            margin: 0 0 10px 0;
            color: #0066cc;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .model-result {
            border: 1px solid #ddd;
            padding: 15px;
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
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .confidence {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .confidence.high { background: #4caf50; color: white; }
        .confidence.medium { background: #ff9800; color: white; }
        .confidence.low { background: #f44336; color: white; }
        .description {
            margin-top: 10px;
            color: #333;
            line-height: 1.6;
        }
        .ocr-text {
            margin-top: 10px;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }
        .processing-time {
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .consensus {
            margin-top: 20px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 5px;
            border: 1px solid #2196f3;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
    <h1>🔍 Comprehensive VLM Comparison Report</h1>
    <div class="comparison-container">
"""
    
    # Add summary section
    total_tests = len(results)
    successful_models = set()
    total_processing_time = 0
    
    for result in results:
        for model_name, model_result in result.get("model_comparisons", {}).items():
            if model_result.get("success", False):
                successful_models.add(model_name)
            total_processing_time += model_result.get("processing_time_seconds", 0)
    
    html_content += f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_tests}</div>
                    <div class="stat-label">Images Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(successful_models)}</div>
                    <div class="stat-label">Models Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_processing_time:.1f}s</div>
                    <div class="stat-label">Total Processing Time</div>
                </div>
            </div>
            <p style="margin-top: 20px;"><strong>Models:</strong> {', '.join(sorted(successful_models))}</p>
        </div>
"""
    
    # Add individual comparisons
    for result in results:
        image_name = result.get("image_name", "Unknown")
        page = result.get("page", 0)
        consensus = result.get("consensus", {})
        best_model = consensus.get("best_model")
        
        # Get image path
        image_path = None
        for f in (log_dir.parent / "visual_elements_with_vlm" / "extracted_images").glob("*.png"):
            if image_name.lower() in f.name.lower() or f"page{page}" in f.name:
                image_path = f
                break
        
        html_content += f"""
        <div class="image-comparison">
            <div class="image-header">
"""
        
        if image_path and image_path.exists():
            html_content += f'<img src="{image_path.absolute()}" class="image-thumbnail" alt="{image_name}">'
        
        html_content += f"""
                <div class="image-info">
                    <h2>{image_name}</h2>
                    <p>Page: {page}</p>
                    <p>Generated: {result.get('timestamp', 'N/A')}</p>
                </div>
            </div>
            
            <div class="models-grid">
"""
        
        # Add model results
        for model_name, model_result in result.get("model_comparisons", {}).items():
            is_best = model_name == best_model
            confidence = model_result.get("confidence", 0)
            confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
            
            html_content += f"""
                <div class="model-result {'best' if is_best else ''}">
                    <div class="model-header">{model_name} {'⭐' if is_best else ''}</div>
                    <span class="confidence {confidence_class}">
                        Confidence: {confidence:.1%}
                    </span>
                    <div class="processing-time">
                        Processing: {model_result.get('processing_time_seconds', 0):.2f}s
                    </div>
                    <div class="description">
                        {model_result.get('description', 'No description available')[:300]}...
                    </div>
"""
            
            if model_result.get('extracted_text'):
                html_content += f"""
                    <div class="ocr-text">
                        <strong>OCR:</strong> {model_result.get('extracted_text', '')[:200]}...
                    </div>
"""
            
            html_content += "</div>"
        
        # Add consensus section
        html_content += f"""
            </div>
            <div class="consensus">
                <strong>Consensus Score:</strong> {consensus.get('score', 0):.2f} | 
                <strong>Best Model:</strong> {best_model or 'None'}
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "all_vlms_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {report_path}")
    return report_path

async def main():
    """Run comprehensive VLM comparison"""
    
    # Define which models to test
    # Start with smaller models for testing
    test_models = [
        VLMModelType.QWEN25_VL_7B,      # Works with Transformers fallback
        VLMModelType.LLAMA32_VISION_11B, # Works with vLLM
        VLMModelType.LLAVA_16_7B,        # Smaller LLaVA variant
        # VLMModelType.PIXTRAL_12B,      # Enable if you want to test Pixtral
    ]
    
    logger.info(f"Initializing Multi-VLM comparison with {len(test_models)} models...")
    
    # Initialize multi-VLM integration
    multi_vlm = MultiVLMIntegration(enabled_models=test_models)
    
    # Load test images
    visual_elements = load_test_images()
    logger.info(f"Loaded {len(visual_elements)} test images")
    
    # Run comparative analysis
    results = await multi_vlm.analyze_visual_elements_comparative(visual_elements)
    
    # Save JSON results
    json_results = []
    for element, comparison in results:
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
    
    results_file = log_dir / "all_vlms_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(json_results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    for result in json_results:
        logger.info(f"\nImage: {result['image_name']}")
        logger.info(f"Best Model: {result['consensus']['best_model']}")
        logger.info(f"Consensus Score: {result['consensus']['score']:.2f}")
        
        for model, data in result['model_comparisons'].items():
            logger.info(f"  - {model}: {data['confidence']:.1%} confidence, {data['processing_time_seconds']:.2f}s")
    
    logger.info("\n✅ Comparison complete!")
    logger.info(f"📊 View report at: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())