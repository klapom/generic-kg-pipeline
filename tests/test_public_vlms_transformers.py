#!/usr/bin/env python3
"""
Public VLMs Test with Transformers

Tests publicly available VLMs using Transformers clients:
- Qwen2.5-VL-7B
- LLaVA-1.6-Mistral-7B  
- Pixtral-12B
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/public_vlms_transformers")
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

from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType

def test_vlm(client, image_data: bytes, model_name: str) -> Dict[str, Any]:
    """Test a single VLM client"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    try:
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="description"
        )
        
        logger.info(f"Success: {result.success}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Processing Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"Description: {result.description[:200]}...")
        if result.ocr_text:
            logger.info(f"OCR Text: {result.ocr_text[:100]}...")
        
        return {
            "model": model_name,
            "success": result.success,
            "confidence": result.confidence,
            "processing_time": result.processing_time_seconds,
            "description": result.description,
            "ocr_text": result.ocr_text,
            "error_message": result.error_message
        }
        
    except Exception as e:
        logger.error(f"Failed to test {model_name}: {e}")
        return {
            "model": model_name,
            "success": False,
            "error_message": str(e)
        }
    finally:
        # Cleanup
        if hasattr(client, 'cleanup'):
            client.cleanup()

def generate_comparison_html(results: List[Dict[str, Any]], output_dir: Path):
    """Generate HTML comparison report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Public VLMs Comparison - Transformers</title>
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
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .model-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .model-card.success {
            border-color: #4caf50;
        }
        .model-card.failed {
            border-color: #f44336;
        }
        .model-header {
            font-size: 1.2em;
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 15px;
        }
        .metric {
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
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
            margin-top: 15px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
            line-height: 1.6;
        }
        .ocr-text {
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }
        .error {
            color: #f44336;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>🔍 Public VLMs Comparison - Transformers Backend</h1>
"""
    
    # Summary
    successful = [r for r in results if r.get('success', False)]
    total_time = sum(r.get('processing_time', 0) for r in results)
    
    html_content += f"""
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Models Tested:</strong> {len(results)}</p>
        <p><strong>Successful:</strong> {len(successful)}</p>
        <p><strong>Total Processing Time:</strong> {total_time:.2f}s</p>
        <p><strong>Test Image:</strong> BMW front view with annotations</p>
    </div>
    
    <div class="models-grid">
"""
    
    # Individual model results
    for result in results:
        success = result.get('success', False)
        confidence = result.get('confidence', 0)
        confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
        
        html_content += f"""
        <div class="model-card {'success' if success else 'failed'}">
            <div class="model-header">{result['model']}</div>
"""
        
        if success:
            html_content += f"""
            <div class="metric">
                <span class="metric-label">Status:</span>
                <span style="color: #4caf50;">✅ Success</span>
            </div>
            <div class="metric">
                <span class="metric-label">Confidence:</span>
                <span class="confidence {confidence_class}">{confidence:.1%}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Processing Time:</span>
                <span>{result.get('processing_time', 0):.2f}s</span>
            </div>
            <div class="description">
                <strong>Description:</strong><br>
                {result.get('description', 'No description')}
            </div>
"""
            if result.get('ocr_text'):
                html_content += f"""
            <div class="ocr-text">
                <strong>OCR Text:</strong><br>
                {result.get('ocr_text', '')}
            </div>
"""
        else:
            html_content += f"""
            <div class="metric">
                <span class="metric-label">Status:</span>
                <span style="color: #f44336;">❌ Failed</span>
            </div>
            <div class="error">
                {result.get('error_message', 'Unknown error')}
            </div>
"""
        
        html_content += "</div>"
    
    html_content += """
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "public_vlms_transformers_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {report_path}")
    return report_path

def main():
    """Run public VLMs comparison"""
    
    # Load test image
    test_image_path = Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png")
    
    if not test_image_path.exists():
        logger.error(f"Test image not found: {test_image_path}")
        return
        
    logger.info(f"Loading test image: {test_image_path}")
    
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    results = []
    
    # Test Qwen2.5-VL
    logger.info("\nInitializing Qwen2.5-VL-7B...")
    qwen_client = TransformersQwen25VLClient(
        temperature=0.1,
        max_new_tokens=1024
    )
    results.append(test_vlm(qwen_client, image_data, "Qwen2.5-VL-7B"))
    
    # Test LLaVA
    logger.info("\nInitializing LLaVA-1.6-Mistral-7B...")
    llava_client = TransformersLLaVAClient(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_8bit=True,
        temperature=0.1,
        max_new_tokens=1024
    )
    results.append(test_vlm(llava_client, image_data, "LLaVA-1.6-Mistral-7B"))
    
    # Test Pixtral
    logger.info("\nInitializing Pixtral-12B...")
    pixtral_client = TransformersPixtralClient(
        temperature=0.1,
        max_new_tokens=1024,
        load_in_8bit=True
    )
    results.append(test_vlm(pixtral_client, image_data, "Pixtral-12B"))
    
    # Save JSON results
    results_file = log_dir / "public_vlms_transformers_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    for result in results:
        if result.get('success'):
            logger.info(f"\n✅ {result['model']}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.1%}")
            logger.info(f"   Processing: {result.get('processing_time', 0):.2f}s")
        else:
            logger.info(f"\n❌ {result['model']}")
            logger.info(f"   Error: {result.get('error_message', 'Unknown')}")
    
    logger.info(f"\n📊 View report at: {report_path}")
    logger.info("✅ Comparison complete!")

if __name__ == "__main__":
    main()