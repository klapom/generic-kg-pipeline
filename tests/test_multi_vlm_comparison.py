#!/usr/bin/env python3
"""Test multi-VLM comparison with Qwen2.5-VL and Llama-3.2-11B-Vision"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
log_dir = Path("tests/debugging/multi_vlm_comparison")
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

# Import our modules
import sys
sys.path.insert(0, '.')

from core.parsers.vlm_integration import MultiVLMIntegration, VLMModelType
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

async def compare_vlm_models():
    """Compare multiple VLM models on BMW images"""
    
    # Initialize multi-VLM with both models
    logger.info("Initializing Multi-VLM comparison with Qwen2.5-VL and Llama-3.2-11B-Vision...")
    multi_vlm = MultiVLMIntegration(
        enabled_models=[
            VLMModelType.QWEN25_VL_7B,
            VLMModelType.LLAMA32_VISION_11B
        ]
    )
    
    # Select test images
    test_images = [
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
            "name": "BMW front view with annotations",
            "page": 3
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
            "name": "BMW interior dashboard",
            "page": 5
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element1.png"),
            "name": "BMW interior seats",
            "page": 5
        }
    ]
    
    # Prepare visual elements
    visual_elements = []
    for img_info in test_images:
        if not img_info["path"].exists():
            logger.warning(f"Image not found: {img_info['path']}")
            continue
            
        logger.info(f"Loading image: {img_info['name']}")
        
        # Load image bytes
        with open(img_info["path"], 'rb') as f:
            image_bytes = f.read()
        
        # Create VisualElement
        visual_element = VisualElement(
            element_type=VisualElementType.IMAGE,
            source_format=DocumentType.PDF,
            content_hash=VisualElement.create_hash(image_bytes),
            page_or_slide=img_info["page"],
            raw_data=image_bytes,
            analysis_metadata={"name": img_info["name"]}
        )
        
        visual_elements.append(visual_element)
    
    logger.info(f"Prepared {len(visual_elements)} visual elements for comparison")
    
    # Perform comparative analysis
    comparison_results = await multi_vlm.analyze_visual_elements_comparative(
        visual_elements,
        document_context={
            'document_type': 'BMW 3 Series G20 brochure',
            'document_title': 'BMW 3 Series Preview'
        },
        max_elements=len(visual_elements)  # Analyze all
    )
    
    # Process and save results
    results_summary = []
    
    for element in visual_elements:
        if element.content_hash in comparison_results:
            comparison = comparison_results[element.content_hash]
            
            result = {
                "image_name": element.analysis_metadata.get("name", "Unknown"),
                "page": element.page_or_slide,
                "model_comparisons": comparison.model_results,
                "consensus": {
                    "score": comparison.consensus_score,
                    "best_model": comparison.best_model
                },
                "timestamp": comparison.timestamp
            }
            
            results_summary.append(result)
            
            # Log comparison
            logger.info(f"\n🎯 Image: {result['image_name']}")
            for model_name, model_result in comparison.model_results.items():
                logger.info(f"  {model_name}:")
                logger.info(f"    - Confidence: {model_result['confidence']:.2%}")
                logger.info(f"    - Type: {model_result['visual_subtype']}")
                logger.info(f"    - Processing time: {model_result['processing_time_seconds']:.2f}s")
                logger.info(f"    - Description preview: {model_result['description'][:100]}...")
    
    # Save comparison results
    results_file = log_dir / "multi_vlm_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # Create HTML comparison report
    create_comparison_report(results_summary, test_images, log_dir)
    
    logger.info(f"✅ Multi-VLM comparison completed. Results saved to {log_dir}")
    return results_summary

def create_comparison_report(results, test_images, output_dir):
    """Create an HTML report comparing multiple VLM outputs"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Multi-VLM Comparison Report</title>
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
            width: 150px;
            height: 150px;
            object-fit: cover;
            border: 1px solid #ddd;
            margin-right: 20px;
        }
        .image-info h2 {
            margin: 0 0 10px 0;
            color: #0066cc;
        }
        .image-info p {
            margin: 5px 0;
            color: #666;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .model-result {
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
        }
        .model-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .model-name {
            font-size: 18px;
            color: #0066cc;
        }
        .confidence {
            font-size: 24px;
            font-weight: bold;
            float: right;
        }
        .confidence.high { color: #28a745; }
        .confidence.medium { color: #ffc107; }
        .confidence.low { color: #dc3545; }
        .metrics {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            font-size: 14px;
        }
        .description {
            background: white;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            font-size: 14px;
            line-height: 1.5;
            max-height: 200px;
            overflow-y: auto;
        }
        .consensus {
            background: #e8f4f8;
            border: 2px solid #0066cc;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
        }
        .consensus h3 {
            margin: 0 0 10px 0;
            color: #0066cc;
        }
        .best-model {
            font-size: 18px;
            color: #28a745;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Multi-VLM Comparison: Qwen2.5-VL vs Llama-3.2-11B-Vision</h1>
    <p style="text-align: center;">Comparative analysis of visual elements from BMW 3 Series G20 document</p>
    
    <div class="comparison-container">
"""
    
    # Create image path mapping
    image_paths = {img["name"]: img["path"] for img in test_images}
    
    for result in results:
        # Get confidence class
        def get_confidence_class(conf):
            if conf >= 0.8:
                return "high"
            elif conf >= 0.5:
                return "medium"
            else:
                return "low"
        
        # Find image path
        image_name = result["image_name"]
        image_path = image_paths.get(image_name)
        image_src = f"../visual_elements_with_vlm/extracted_images/{image_path.name}" if image_path else ""
        
        html_content += f"""
        <div class="image-comparison">
            <div class="image-header">
                <img src="{image_src}" alt="{image_name}" class="image-thumbnail">
                <div class="image-info">
                    <h2>{image_name}</h2>
                    <p>Page: {result['page']}</p>
                    <p>Analysis timestamp: {result['timestamp']}</p>
                </div>
            </div>
            
            <div class="models-grid">
"""
        
        # Add each model's results
        for model_name, model_result in result["model_comparisons"].items():
            confidence = model_result["confidence"]
            conf_class = get_confidence_class(confidence)
            
            html_content += f"""
                <div class="model-result">
                    <div class="model-header">
                        <span class="model-name">{model_name}</span>
                        <span class="confidence {conf_class}">{confidence:.1%}</span>
                    </div>
                    <div class="metrics">
                        <span>Type: {model_result['visual_subtype']}</span>
                        <span>Time: {model_result['processing_time_seconds']:.2f}s</span>
                    </div>
                    <div class="description">
                        {model_result['description']}
                    </div>
                </div>
"""
        
        html_content += f"""
            </div>
            
            <div class="consensus">
                <h3>Consensus Analysis</h3>
                <p>Average Confidence: {result['consensus']['score']:.1%}</p>
                <p class="best-model">Best Model: {result['consensus']['best_model']}</p>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save HTML report
    html_path = output_dir / "multi_vlm_comparison_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML comparison report created: {html_path}")

if __name__ == "__main__":
    asyncio.run(compare_vlm_models())