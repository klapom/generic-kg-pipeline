#!/usr/bin/env python3
"""
Complete VLM Comparison Test with all 3 models

Tests Qwen2.5-VL, LLaVA, and Pixtral on BMW document images.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import List, Dict, Any

# Setup logging
log_dir = Path("tests/debugging/complete_vlm_comparison")
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
        start_time = time.time()
        result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        end_time = time.time()
        
        logger.info(f"Success: {result.success}")
        logger.info(f"Confidence: {result.confidence:.2%}")
        logger.info(f"Processing Time: {end_time - start_time:.2f}s")
        logger.info(f"Description Preview: {result.description[:150]}...")
        if result.ocr_text:
            logger.info(f"OCR Text: {result.ocr_text}")
        if result.extracted_data:
            logger.info(f"Extracted Data: {result.extracted_data}")
        
        return {
            "model": model_name,
            "success": result.success,
            "confidence": result.confidence,
            "processing_time": end_time - start_time,
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


def generate_comparison_html(results: List[Dict[str, Any]], output_dir: Path):
    """Generate side-by-side comparison HTML report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🔥 VLM Comparison - BMW Document Analysis</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        h1 { 
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        @media (max-width: 1200px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }
        .model-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s;
        }
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .model-header {
            padding: 20px;
            font-size: 1.3em;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #f0f0f0;
        }
        .model-header.qwen { background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); }
        .model-header.llava { background: linear-gradient(135deg, #f09333320 0%, #e6683c20 100%); }
        .model-header.pixtral { background: linear-gradient(135deg, #56ccf220 0%, #2f80ed20 100%); }
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
        .model-content {
            padding: 20px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
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
        .description-box {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            line-height: 1.6;
            font-size: 0.95em;
        }
        .ocr-box {
            margin-top: 15px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .data-box {
            margin-top: 15px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        .winner-highlight {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            50% { box-shadow: 0 5px 20px rgba(255,215,0,0.3); }
            100% { box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 VLM Comparison - BMW Document Analysis</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Comparing Qwen2.5-VL vs LLaVA vs Pixtral</p>
    </div>
    
    <div class="container">
"""
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    total_time = sum(r.get('processing_time', 0) for r in results)
    avg_confidence = sum(r.get('confidence', 0) for r in successful) / len(successful) if successful else 0
    
    # Find best model
    best_model = None
    if successful:
        best_model = max(successful, key=lambda x: x.get('confidence', 0))
    
    html_content += f"""
        <div class="summary-card">
            <h2>📊 Test Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Models Tested</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(successful)}/{len(results)}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{total_time:.1f}s</div>
                    <div class="stat-label">Total Time</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_confidence:.1%}</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
        </div>
        
        <div class="comparison-grid">
"""
    
    # Model cards
    model_classes = {
        "Qwen2.5-VL-7B": "qwen",
        "LLaVA-1.6-Mistral-7B": "llava",
        "Pixtral-12B": "pixtral"
    }
    
    for result in results:
        model_name = result['model']
        success = result.get('success', False)
        confidence = result.get('confidence', 0)
        confidence_class = "high" if confidence > 0.8 else ("medium" if confidence > 0.6 else "low")
        is_best = best_model and result['model'] == best_model['model']
        model_class = model_classes.get(model_name, "")
        
        html_content += f"""
            <div class="model-card {'winner-highlight' if is_best else ''}">
                <div class="model-header {model_class}">
                    <span>{model_name} {'👑' if is_best else ''}</span>
                    <span class="status-badge {'status-success' if success else 'status-failed'}">
                        {'✅ Success' if success else '❌ Failed'}
                    </span>
                </div>
                <div class="model-content">
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
                    
                    <div class="description-box">
                        <strong>📝 Description:</strong><br>
                        {result.get('description', 'No description')}
                    </div>
"""
            if result.get('ocr_text'):
                html_content += f"""
                    <div class="ocr-box">
                        <strong>🔤 OCR Text:</strong><br>
                        {result.get('ocr_text', '').replace(chr(10), '<br>')}
                    </div>
"""
            if result.get('extracted_data'):
                html_content += f"""
                    <div class="data-box">
                        <strong>📊 Extracted Data:</strong><br>
                        {json.dumps(result.get('extracted_data', {}), indent=2).replace(chr(10), '<br>').replace(' ', '&nbsp;')}
                    </div>
"""
        else:
            html_content += f"""
                    <div style="color: #dc3545; font-style: italic; padding: 20px;">
                        <strong>❌ Error:</strong> {result.get('error_message', 'Unknown error')}
                    </div>
"""
        
        html_content += """
                </div>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "complete_vlm_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report saved to: {report_path}")
    return report_path


def main():
    """Run complete VLM comparison on BMW document"""
    
    logger.info("Starting Complete VLM Comparison Test")
    logger.info("=" * 80)
    
    # Use BMW document images
    test_images = [
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
            "name": "BMW front view with annotations"
        },
        {
            "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"), 
            "name": "BMW interior view"
        }
    ]
    
    # Find first available image
    image_path = None
    image_name = None
    for img_info in test_images:
        if img_info["path"].exists():
            image_path = img_info["path"]
            image_name = img_info["name"]
            break
    
    if not image_path:
        # Fallback to any available BMW image
        available_images = list(Path("tests/debugging/visual_elements_with_vlm/extracted_images").glob("*.png"))
        if available_images:
            image_path = available_images[0]
            image_name = f"BMW image ({image_path.name})"
        else:
            logger.error("No test images found!")
            return
    
    logger.info(f"🖼️  Using test image: {image_name}")
    logger.info(f"📁 Path: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    results = []
    
    # Test 1: Qwen2.5-VL
    logger.info("\n🚀 Starting Qwen2.5-VL test...")
    qwen_client = TransformersQwen25VLClient(
        temperature=0.2,
        max_new_tokens=512
    )
    result = test_vlm(qwen_client, image_data, "Qwen2.5-VL-7B")
    results.append(result)
    
    # Test 2: LLaVA
    logger.info("\n🚀 Starting LLaVA test...")
    llava_client = TransformersLLaVAClient(
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_8bit=True,
        temperature=0.2,
        max_new_tokens=512
    )
    result = test_vlm(llava_client, image_data, "LLaVA-1.6-Mistral-7B")
    results.append(result)
    
    # Test 3: Pixtral (with extended timeout)
    logger.info("\n🚀 Starting Pixtral test...")
    logger.info("⏳ Note: Pixtral may take longer to load on first run...")
    try:
        pixtral_client = TransformersPixtralClient(
            temperature=0.2,
            max_new_tokens=512,
            load_in_8bit=True
        )
        result = test_vlm(pixtral_client, image_data, "Pixtral-12B")
        results.append(result)
    except Exception as e:
        logger.error(f"Pixtral initialization failed: {e}")
        results.append({
            "model": "Pixtral-12B",
            "success": False,
            "error_message": f"Initialization failed: {e}",
            "processing_time": 0,
            "confidence": 0
        })
    
    # Save JSON results
    results_file = log_dir / "complete_vlm_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🏆 COMPARISON RESULTS")
    logger.info("="*80)
    
    for result in results:
        status = "✅" if result.get('success') else "❌"
        logger.info(f"\n{status} {result['model']}:")
        if result.get('success'):
            logger.info(f"   • Confidence: {result.get('confidence', 0):.1%}")
            logger.info(f"   • Time: {result.get('processing_time', 0):.2f}s")
            logger.info(f"   • OCR: {'Yes' if result.get('ocr_text') else 'No'}")
            logger.info(f"   • Structured Data: {'Yes' if result.get('extracted_data') else 'No'}")
        else:
            logger.info(f"   • Error: {result.get('error_message', 'Unknown error')}")
    
    # Find winner
    successful = [r for r in results if r.get('success')]
    if successful:
        best = max(successful, key=lambda x: x.get('confidence', 0))
        logger.info(f"\n👑 Best Model: {best['model']} ({best.get('confidence', 0):.1%} confidence)")
    
    logger.info(f"\n📊 View detailed comparison at: {report_path}")
    logger.info("\n🎉 VLM comparison complete!")
    
    # Return paths for user
    return {
        "log_file": str(log_dir / f'comparison_{datetime.now():%Y%m%d_%H%M%S}.log'),
        "json_results": str(results_file),
        "html_report": str(report_path)
    }


if __name__ == "__main__":
    paths = main()
    print(f"\n📍 All results available at:")
    for key, path in paths.items():
        print(f"   • {key}: {path}")