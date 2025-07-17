#!/usr/bin/env python3
"""
Final VLM Comparison Test

Comprehensive test of all available VLMs:
- Qwen2.5-VL-7B (fixed)
- LLaVA-1.6-Mistral-7B (fixed) 
- Pixtral-12B (new)
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/final_vlm_comparison")
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
    """Test a single VLM client with error handling"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    try:
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        logger.info(f"Success: {result.success}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Processing Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"Description Preview: {result.description[:150]}...")
        if result.ocr_text:
            logger.info(f"OCR Text: {result.ocr_text}")
        if result.extracted_data:
            logger.info(f"Extracted Data: {result.extracted_data}")
        
        return {
            "model": model_name,
            "success": result.success,
            "confidence": result.confidence,
            "processing_time": result.processing_time_seconds,
            "description": result.description,
            "ocr_text": result.ocr_text,
            "extracted_data": result.extracted_data,
            "error_message": result.error_message,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to test {model_name}: {e}")
        return {
            "model": model_name,
            "success": False,
            "error_message": str(e),
            "processing_time": 0,
            "confidence": 0
        }
    finally:
        # Always cleanup
        if hasattr(client, 'cleanup'):
            try:
                client.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for {model_name}: {e}")

def generate_comprehensive_html(results: List[Dict[str, Any]], output_dir: Path):
    """Generate comprehensive HTML comparison report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🔥 Final VLM Comparison - All Models</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #2c3e50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .summary h2 {
            margin-top: 0;
            font-size: 1.8em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .model-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .model-card:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .model-card.success:before {
            background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        }
        .model-card.failed:before {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
        }
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .model-header {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .status-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 12px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .metric-label {
            font-weight: 600;
            color: #6c757d;
        }
        .confidence {
            padding: 4px 10px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .confidence.high { background: #d4edda; color: #155724; }
        .confidence.medium { background: #fff3cd; color: #856404; }
        .confidence.low { background: #f8d7da; color: #721c24; }
        .description {
            margin-top: 20px;
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            line-height: 1.7;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .ocr-text {
            margin-top: 15px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .extracted-data {
            margin-top: 15px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .error-message {
            color: #dc3545;
            font-style: italic;
            background: #f8d7da;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .winner-crown {
            font-size: 1.5em;
            color: #ffd700;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 Final VLM Comparison Report</h1>
"""
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    total_time = sum(r.get('processing_time', 0) for r in results)
    
    # Find best model by confidence
    best_model = None
    if successful:
        best_model = max(successful, key=lambda x: x.get('confidence', 0))
    
    html_content += f"""
        <div class="summary">
            <h2>📊 Test Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-value">{len(results)}</span>
                    <span class="stat-label">Models Tested</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{len(successful)}</span>
                    <span class="stat-label">Successful</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{total_time:.1f}s</span>
                    <span class="stat-label">Total Time</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">{best_model['model'].split('-')[0] if best_model else 'None'}</span>
                    <span class="stat-label">Best Model</span>
                </div>
            </div>
        </div>
        
        <div class="models-grid">
"""
    
    # Sort results by success and confidence
    sorted_results = sorted(results, key=lambda x: (x.get('success', False), x.get('confidence', 0)), reverse=True)
    
    for i, result in enumerate(sorted_results):
        success = result.get('success', False)
        confidence = result.get('confidence', 0)
        confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
        is_best = best_model and result['model'] == best_model['model']
        
        html_content += f"""
            <div class="model-card {'success' if success else 'failed'}">
                <div class="model-header">
                    <span>{result['model']} {'👑' if is_best else ''}</span>
                    <span class="status-badge {'status-success' if success else 'status-failed'}">
                        {'✅ Success' if success else '❌ Failed'}
                    </span>
                </div>
"""
        
        if success:
            html_content += f"""
                <div class="metric-row">
                    <span class="metric-label">Confidence:</span>
                    <span class="confidence {confidence_class}">{confidence:.1%}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Processing Time:</span>
                    <span>{result.get('processing_time', 0):.2f}s</span>
                </div>
                <div class="description">
                    <strong>📝 Description:</strong><br>
                    {result.get('description', 'No description')}
                </div>
"""
            if result.get('ocr_text'):
                html_content += f"""
                <div class="ocr-text">
                    <strong>🔤 OCR Text:</strong><br>
                    {result.get('ocr_text', '')}
                </div>
"""
            if result.get('extracted_data'):
                html_content += f"""
                <div class="extracted-data">
                    <strong>📊 Extracted Data:</strong><br>
                    {json.dumps(result.get('extracted_data', {}), indent=2)}
                </div>
"""
        else:
            html_content += f"""
                <div class="error-message">
                    <strong>❌ Error:</strong> {result.get('error_message', 'Unknown error')}
                </div>
"""
        
        html_content += "</div>"
    
    html_content += """
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "final_vlm_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report saved to: {report_path}")
    return report_path

def main():
    """Run final VLM comparison"""
    
    # Test with multiple images
    test_images = [
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
            "name": "BMW front view"
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
            "name": "BMW interior"
        }
    ]
    
    all_results = []
    
    for img_info in test_images:
        if not img_info["path"].exists():
            logger.warning(f"Test image not found: {img_info['path']}")
            continue
            
        logger.info(f"\n{'#'*80}")
        logger.info(f"🖼️  Testing with image: {img_info['name']}")
        logger.info('#'*80)
        
        with open(img_info["path"], 'rb') as f:
            image_data = f.read()
        
        results = []
        
        # Test 1: Qwen2.5-VL (known to work)
        logger.info("\n🔧 Initializing Qwen2.5-VL-7B...")
        qwen_client = TransformersQwen25VLClient(
            temperature=0.2,
            max_new_tokens=512
        )
        result = test_vlm(qwen_client, image_data, "Qwen2.5-VL-7B")
        result["image"] = img_info["name"]
        results.append(result)
        
        # Test 2: LLaVA (fixed)
        logger.info("\n🔧 Initializing LLaVA-1.6-Mistral-7B...")
        llava_client = TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.2,
            max_new_tokens=512
        )
        result = test_vlm(llava_client, image_data, "LLaVA-1.6-Mistral-7B")
        result["image"] = img_info["name"]
        results.append(result)
        
        # Test 3: Pixtral (if available)
        try:
            logger.info("\n🔧 Initializing Pixtral-12B...")
            pixtral_client = TransformersPixtralClient(
                temperature=0.2,
                max_new_tokens=512,
                load_in_8bit=True
            )
            result = test_vlm(pixtral_client, image_data, "Pixtral-12B")
            result["image"] = img_info["name"]
            results.append(result)
        except Exception as e:
            logger.warning(f"⚠️  Pixtral test skipped: {e}")
            results.append({
                "model": "Pixtral-12B",
                "image": img_info["name"],
                "success": False,
                "error_message": f"Model not available: {e}",
                "processing_time": 0,
                "confidence": 0
            })
        
        all_results.extend(results)
        
        # Break after first image to save time
        break
    
    # Save JSON results
    results_file = log_dir / "final_vlm_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comprehensive_html(all_results, log_dir)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL COMPARISON SUMMARY")
    logger.info("="*80)
    
    successful = [r for r in all_results if r.get('success')]
    failed = [r for r in all_results if not r.get('success')]
    
    logger.info(f"\n✅ Successful Models: {len(successful)}")
    for result in successful:
        logger.info(f"   • {result['model']}: {result.get('confidence', 0):.1%} confidence, {result.get('processing_time', 0):.2f}s")
    
    if failed:
        logger.info(f"\n❌ Failed Models: {len(failed)}")
        for result in failed:
            logger.info(f"   • {result['model']}: {result.get('error_message', 'Unknown error')}")
    
    if successful:
        best = max(successful, key=lambda x: x.get('confidence', 0))
        logger.info(f"\n👑 Best Model: {best['model']} ({best.get('confidence', 0):.1%} confidence)")
    
    logger.info(f"\n📊 View detailed report at: {report_path}")
    logger.info("🎉 Final VLM comparison complete!")

if __name__ == "__main__":
    main()