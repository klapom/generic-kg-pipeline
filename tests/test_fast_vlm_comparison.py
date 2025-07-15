#!/usr/bin/env python3
"""
Fast VLM Comparison using cached models

This version ensures models are loaded from local cache for faster performance.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF
import os

# Set HuggingFace to offline mode to ensure local cache is used
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Setup logging
log_dir = Path("tests/debugging/fast_vlm_comparison")
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

from core.utils.model_cache_manager import ModelCacheManager
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType


def verify_cache():
    """Verify all models are cached before starting"""
    manager = ModelCacheManager()
    
    logger.info("🔍 Verifying model cache...")
    all_cached = True
    
    for model_name in manager.MODEL_SPECS:
        info = manager.get_cache_info(model_name)
        if info["cached"]:
            logger.info(f"✅ {model_name}: {info['size_gb']:.1f}GB cached")
        else:
            logger.error(f"❌ {model_name}: NOT CACHED!")
            all_cached = False
            
    if not all_cached:
        logger.error("Some models are not cached. Run with online mode to download.")
        # Disable offline mode for downloading
        del os.environ["HF_HUB_OFFLINE"]
        del os.environ["TRANSFORMERS_OFFLINE"]
        logger.info("Switched to online mode for downloading...")
    else:
        logger.info("✅ All models cached - using offline mode for speed!")
        
    return all_cached


def test_vlm_timed(client, image_data: bytes, model_name: str) -> Dict[str, Any]:
    """Test a single VLM with detailed timing"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_name}")
    logger.info('='*60)
    
    result = {
        "model": model_name,
        "timings": {}
    }
    
    try:
        # Time the inference
        inference_start = time.time()
        
        analysis_result = client.analyze_visual(
            image_data=image_data,
            element_type=VisualElementType.IMAGE,
            analysis_focus="comprehensive"
        )
        
        inference_time = time.time() - inference_start
        result["timings"]["inference"] = inference_time
        
        logger.info(f"Success: {analysis_result.success}")
        logger.info(f"Confidence: {analysis_result.confidence:.2%}")
        logger.info(f"Inference Time: {inference_time:.2f}s")
        
        if analysis_result.ocr_text:
            logger.info(f"OCR Text: {analysis_result.ocr_text}")
        if analysis_result.extracted_data:
            logger.info(f"Extracted Data: {analysis_result.extracted_data}")
        
        result.update({
            "success": analysis_result.success,
            "confidence": analysis_result.confidence,
            "description": analysis_result.description,
            "ocr_text": analysis_result.ocr_text,
            "extracted_data": analysis_result.extracted_data,
            "error_message": analysis_result.error_message
        })
        
    except Exception as e:
        logger.error(f"Failed to test {model_name}: {e}")
        result.update({
            "success": False,
            "error_message": str(e),
            "confidence": 0
        })
    finally:
        # Cleanup
        cleanup_start = time.time()
        if hasattr(client, 'cleanup'):
            try:
                client.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for {model_name}: {e}")
        result["timings"]["cleanup"] = time.time() - cleanup_start
        
    return result


def generate_performance_report(all_results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Generate performance-focused HTML report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>⚡ Fast VLM Performance Comparison</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f0f2f5;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
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
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .performance-summary {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .timing-chart {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .timing-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .timing-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .timing-label {
            color: #7f8c8d;
            margin-top: 5px;
            font-size: 1.1em;
        }
        .model-comparison {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .best-value {
            background: #d4edda;
            font-weight: bold;
        }
        .cache-info {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #4caf50;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
        }
        @media (max-width: 1200px) {
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
        .result-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-bar {
            background: #ecf0f1;
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .metric-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ Fast VLM Performance Comparison</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Using Cached Models for Maximum Speed</p>
    </div>
    
    <div class="container">
"""
    
    # Calculate overall statistics
    total_tests = sum(len(results) for results in all_results.values())
    model_stats = {}
    
    for image_path, results in all_results.items():
        for result in results:
            model = result['model']
            if model not in model_stats:
                model_stats[model] = {
                    'inference_times': [],
                    'successes': 0,
                    'total': 0,
                    'confidences': []
                }
            
            model_stats[model]['total'] += 1
            if result.get('success'):
                model_stats[model]['successes'] += 1
                model_stats[model]['confidences'].append(result.get('confidence', 0))
                if 'timings' in result and 'inference' in result['timings']:
                    model_stats[model]['inference_times'].append(result['timings']['inference'])
    
    # Cache info
    html_content += """
        <div class="cache-info">
            <h3>🚀 Cache Status</h3>
            <p>All models loaded from local cache for maximum performance!</p>
            <p>Cache location: ~/.cache/huggingface/hub</p>
        </div>
"""
    
    # Performance summary
    html_content += """
        <div class="performance-summary">
            <h2>⚡ Performance Summary</h2>
            <div class="timing-chart">
"""
    
    # Find best values for highlighting
    best_speed = float('inf')
    best_confidence = 0
    
    for model, stats in model_stats.items():
        if stats['inference_times']:
            avg_time = sum(stats['inference_times']) / len(stats['inference_times'])
            if avg_time < best_speed:
                best_speed = avg_time
        if stats['confidences']:
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            if avg_conf > best_confidence:
                best_confidence = avg_conf
    
    # Model timing cards
    for model, stats in model_stats.items():
        if stats['inference_times']:
            avg_time = sum(stats['inference_times']) / len(stats['inference_times'])
            is_fastest = avg_time == best_speed
            
            html_content += f"""
                <div class="timing-card">
                    <div class="timing-value">{avg_time:.1f}s</div>
                    <div class="timing-label">{model.split('-')[0]} {'⚡' if is_fastest else ''}</div>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="model-comparison">
            <h2>📊 Detailed Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Avg Inference Time</th>
                    <th>Success Rate</th>
                    <th>Avg Confidence</th>
                    <th>OCR Capability</th>
                </tr>
"""
    
    # Table rows
    for model, stats in model_stats.items():
        success_rate = stats['successes'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_time = sum(stats['inference_times']) / len(stats['inference_times']) if stats['inference_times'] else 0
        avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
        
        # Check OCR capability
        has_ocr = False
        for _, results in all_results.items():
            for r in results:
                if r['model'] == model and r.get('ocr_text'):
                    has_ocr = True
                    break
        
        time_class = "best-value" if avg_time == best_speed and avg_time > 0 else ""
        conf_class = "best-value" if avg_conf == best_confidence and avg_conf > 0 else ""
        
        html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td class="{time_class}">{avg_time:.1f}s</td>
                    <td>{success_rate:.0f}%</td>
                    <td class="{conf_class}">{avg_conf:.1%}</td>
                    <td>{'✅ Yes' if has_ocr else '❌ No'}</td>
                </tr>
"""
    
    html_content += """
            </table>
        </div>
"""
    
    # Results by image
    for image_path, results in all_results.items():
        image_name = Path(image_path).name
        
        html_content += f"""
        <div class="model-comparison">
            <h3>📸 Results for: {image_name}</h3>
            <div class="results-grid">
"""
        
        for result in results:
            success = result.get('success', False)
            
            html_content += f"""
                <div class="result-card">
                    <h4>{result['model']}</h4>
                    <p><strong>Status:</strong> {'✅ Success' if success else '❌ Failed'}</p>
"""
            
            if success:
                confidence = result.get('confidence', 0)
                inference_time = result.get('timings', {}).get('inference', 0)
                
                html_content += f"""
                    <p><strong>Inference Time:</strong> {inference_time:.2f}s</p>
                    <p><strong>Confidence:</strong> {confidence:.1%}</p>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {confidence*100}%"></div>
                    </div>
"""
                
                if result.get('ocr_text'):
                    html_content += f"""
                    <p><strong>OCR:</strong> {result['ocr_text'][:50]}...</p>
"""
            
            html_content += """
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += f"""
        <div style="text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 0.9em;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    report_path = output_dir / "fast_vlm_performance_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 Performance report saved to: {report_path}")
    return report_path


def main():
    """Run fast VLM comparison using cached models"""
    
    logger.info("⚡ Starting Fast VLM Comparison (Cached Models)")
    logger.info("=" * 80)
    
    # Verify cache
    cache_ok = verify_cache()
    
    # Use existing image or extract new one
    test_image = Path("tests/debugging/complete_vlm_comparison/extracted_images/bmw_page1_img2.png")
    
    if not test_image.exists():
        logger.info("Extracting test image from PDF...")
        pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
        if pdf_path.exists():
            output_dir = test_image.parent
            output_dir.mkdir(exist_ok=True, parents=True)
            
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pix.save(test_image)
            doc.close()
    
    if not test_image.exists():
        logger.error("No test image available!")
        return
        
    logger.info(f"📸 Using test image: {test_image}")
    
    with open(test_image, 'rb') as f:
        image_data = f.read()
    
    # Store results
    all_results = {str(test_image): []}
    
    # Test models with timing
    models_to_test = [
        ("Qwen2.5-VL-7B", TransformersQwen25VLClient),
        ("LLaVA-1.6-Mistral-7B", lambda: TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.2,
            max_new_tokens=512
        )),
        ("Pixtral-12B", lambda: TransformersPixtralClient(
            temperature=0.2,
            max_new_tokens=512,
            load_in_8bit=True
        ))
    ]
    
    for model_name, client_factory in models_to_test:
        logger.info(f"\n🚀 Testing {model_name}...")
        
        # Time model initialization
        init_start = time.time()
        
        try:
            if callable(client_factory):
                client = client_factory()
            else:
                client = client_factory(temperature=0.2, max_new_tokens=512)
                
            init_time = time.time() - init_start
            logger.info(f"Model initialization time: {init_time:.2f}s")
            
            # Test the model
            result = test_vlm_timed(client, image_data, model_name)
            result['timings']['initialization'] = init_time
            
            all_results[str(test_image)].append(result)
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            all_results[str(test_image)].append({
                "model": model_name,
                "success": False,
                "error_message": str(e),
                "timings": {"initialization": time.time() - init_start}
            })
    
    # Save results
    results_file = log_dir / "fast_vlm_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate performance report
    report_path = generate_performance_report(all_results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("⚡ PERFORMANCE SUMMARY")
    logger.info("="*80)
    
    for _, results in all_results.items():
        for result in results:
            if result.get('success'):
                total_time = sum(result.get('timings', {}).values())
                logger.info(f"\n{result['model']}:")
                logger.info(f"  • Total Time: {total_time:.2f}s")
                logger.info(f"  • Init Time: {result['timings'].get('initialization', 0):.2f}s")
                logger.info(f"  • Inference: {result['timings'].get('inference', 0):.2f}s")
                logger.info(f"  • Confidence: {result.get('confidence', 0):.1%}")
    
    logger.info(f"\n📊 Performance report: {report_path}")
    logger.info(f"📁 JSON results: {results_file}")
    logger.info("\n✅ Fast comparison complete!")
    
    return {
        "report": str(report_path),
        "results": str(results_file),
        "log_dir": str(log_dir)
    }


if __name__ == "__main__":
    paths = main()
    if paths:
        print(f"\n📍 Alle Ergebnisse verfügbar unter:")
        for key, path in paths.items():
            print(f"   • {key}: {path}")