#!/usr/bin/env python3
"""Test Qwen2.5-VL using OpenAI-compatible API approach"""

import asyncio
import subprocess
import time
from pathlib import Path
import json
import logging
from datetime import datetime

# Setup logging
log_dir = Path("tests/debugging/qwen25vl_openai_api")
log_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'test_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
sys.path.insert(0, '.')

from core.clients.vllm_qwen25_vl_openai_client import VLLMQwen25VLOpenAIClient, VisualAnalysisResult
from core.parsers.interfaces.data_models import VisualElement, VisualElementType, DocumentType

def start_vllm_server():
    """Start vLLM server in background"""
    cmd = [
        "vllm", "serve", "Qwen/Qwen2.5-VL-7B-Instruct",
        "--port", "8001",  # Different port to avoid conflicts
        "--host", "0.0.0.0",
        "--dtype", "bfloat16",
        "--limit-mm-per-prompt", "image=5",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.7"
    ]
    
    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        logger.info("Waiting for vLLM server to start...")
        time.sleep(30)  # Give it time to load the model
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        raise

async def test_openai_api():
    """Test Qwen2.5-VL with OpenAI API"""
    
    # Start server
    server_process = None
    try:
        server_process = start_vllm_server()
        
        # Initialize client
        logger.info("Initializing OpenAI API client...")
        client = VLLMQwen25VLOpenAIClient(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            auto_start_server=False  # We already started it
        )
        
        # Test images
        test_images = [
            {
                "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
                "name": "BMW front view with annotations",
                "expected_type": VisualElementType.IMAGE
            },
            {
                "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page5_element0.png"),
                "name": "BMW interior dashboard",
                "expected_type": VisualElementType.IMAGE
            }
        ]
        
        results = []
        
        for img_info in test_images:
            if not img_info["path"].exists():
                logger.warning(f"Image not found: {img_info['path']}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {img_info['name']}")
            logger.info(f"{'='*60}")
            
            # Load image
            with open(img_info["path"], 'rb') as f:
                image_bytes = f.read()
            
            # Test different analysis modes
            for analysis_focus in ["comprehensive", "description"]:
                logger.info(f"\nAnalysis focus: {analysis_focus}")
                
                try:
                    result = client.analyze_visual(
                        image_data=image_bytes,
                        element_type=img_info["expected_type"],
                        analysis_focus=analysis_focus
                    )
                    
                    logger.info(f"Success: {result.success}")
                    logger.info(f"Confidence: {result.confidence:.2%}")
                    logger.info(f"Description: {result.description[:200]}...")
                    if result.ocr_text:
                        logger.info(f"OCR Text: {result.ocr_text[:100]}...")
                    
                    results.append({
                        "image": img_info["name"],
                        "analysis_focus": analysis_focus,
                        "success": result.success,
                        "confidence": result.confidence,
                        "description": result.description,
                        "ocr_text": result.ocr_text,
                        "processing_time": result.processing_time_seconds
                    })
                    
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    results.append({
                        "image": img_info["name"],
                        "analysis_focus": analysis_focus,
                        "success": False,
                        "error": str(e)
                    })
        
        # Save results
        results_file = log_dir / "openai_api_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Create summary report
        create_summary_report(results, log_dir)
        
    finally:
        # Stop server
        if server_process:
            logger.info("Stopping vLLM server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()

def create_summary_report(results, output_dir):
    """Create HTML summary report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Qwen2.5-VL OpenAI API Test Results</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { color: #333; }
        .test-result {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #f9f9f9;
        }
        .success { background: #e8f5e9; }
        .failure { background: #ffebee; }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .description {
            background: white;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        code {
            background: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Qwen2.5-VL OpenAI API Test Results</h1>
    <p>Testing the OpenAI-compatible API approach for Qwen2.5-VL with vLLM</p>
    
    <h2>Configuration</h2>
    <ul>
        <li>Model: Qwen/Qwen2.5-VL-7B-Instruct</li>
        <li>API Type: OpenAI-compatible</li>
        <li>Server: vLLM with OpenAI API</li>
        <li>Port: 8001</li>
    </ul>
    
    <h2>Test Results</h2>
"""
    
    for result in results:
        status_class = "success" if result.get("success", False) else "failure"
        
        html_content += f"""
    <div class="test-result {status_class}">
        <h3>{result['image']} - {result['analysis_focus']}</h3>
        <div class="metric"><strong>Success:</strong> {result.get('success', False)}</div>
        <div class="metric"><strong>Confidence:</strong> {result.get('confidence', 0):.2%}</div>
        <div class="metric"><strong>Processing Time:</strong> {result.get('processing_time', 0):.2f}s</div>
        
        {f'<div class="description"><strong>Description:</strong><br>{result.get("description", "N/A")}</div>' if result.get('success') else f'<div class="description"><strong>Error:</strong><br><code>{result.get("error", "Unknown error")}</code></div>'}
    </div>
"""
    
    html_content += """
    <h2>Summary</h2>
    <ul>
        <li>Total Tests: {total}</li>
        <li>Successful: {success}</li>
        <li>Failed: {failed}</li>
        <li>Success Rate: {rate:.1%}</li>
    </ul>
</body>
</html>
""".format(
        total=len(results),
        success=sum(1 for r in results if r.get("success", False)),
        failed=sum(1 for r in results if not r.get("success", False)),
        rate=sum(1 for r in results if r.get("success", False)) / len(results) if results else 0
    )
    
    # Save report
    report_path = output_dir / "openai_api_test_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report created: {report_path}")

if __name__ == "__main__":
    asyncio.run(test_openai_api())