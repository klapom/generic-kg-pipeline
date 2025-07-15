#!/usr/bin/env python3
"""
Extract images from BMW PDF and compare all VLMs
"""

import asyncio
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import List, Dict, Any
import fitz  # PyMuPDF

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


def extract_images_from_pdf(pdf_path: Path, output_dir: Path, max_images: int = 3) -> List[Path]:
    """Extract images from PDF for testing"""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Extracting images from {pdf_path}")
    doc = fitz.open(pdf_path)
    
    extracted_images = []
    image_count = 0
    
    for page_num in range(min(10, len(doc))):  # Check first 10 pages
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            if image_count >= max_images:
                break
                
            # Get image
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            # Skip small images
            if pix.width < 200 or pix.height < 200:
                continue
                
            # Convert to RGB if necessary
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            # Save image
            img_path = output_dir / f"bmw_page{page_num + 1}_img{img_index + 1}.png"
            pix.save(img_path)
            
            extracted_images.append(img_path)
            image_count += 1
            logger.info(f"Extracted: {img_path.name} ({pix.width}x{pix.height})")
            
            if image_count >= max_images:
                break
    
    doc.close()
    return extracted_images


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


def generate_comparison_html(all_results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Generate comprehensive comparison HTML report"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>🔥 Complete VLM Comparison - BMW Document</title>
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
        .image-section {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .image-preview {
            max-width: 400px;
            max-height: 300px;
            margin: 10px auto;
            display: block;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 20px;
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
            max-height: 200px;
            overflow-y: auto;
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
        .overall-summary {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: bold;
        }
        .winner { background: #fffacd; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 Complete VLM Comparison - BMW Document</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">Qwen2.5-VL vs LLaVA vs Pixtral</p>
    </div>
    
    <div class="container">
"""
    
    # Overall summary
    total_tests = sum(len(results) for results in all_results.values())
    total_successful = sum(len([r for r in results if r.get('success')]) for results in all_results.values())
    
    html_content += f"""
        <div class="overall-summary">
            <h2>📊 Overall Summary</h2>
            <p>Total Images Tested: {len(all_results)}</p>
            <p>Total Model Tests: {total_tests}</p>
            <p>Successful Tests: {total_successful}/{total_tests}</p>
            
            <table>
                <tr>
                    <th>Model</th>
                    <th>Success Rate</th>
                    <th>Avg Confidence</th>
                    <th>Avg Time</th>
                    <th>OCR Quality</th>
                </tr>
"""
    
    # Calculate model statistics
    model_stats = {}
    for image_path, results in all_results.items():
        for result in results:
            model = result['model']
            if model not in model_stats:
                model_stats[model] = {
                    'successes': 0,
                    'total': 0,
                    'confidences': [],
                    'times': [],
                    'has_ocr': 0
                }
            model_stats[model]['total'] += 1
            if result.get('success'):
                model_stats[model]['successes'] += 1
                model_stats[model]['confidences'].append(result.get('confidence', 0))
                model_stats[model]['times'].append(result.get('processing_time', 0))
                if result.get('ocr_text'):
                    model_stats[model]['has_ocr'] += 1
    
    # Find best model
    best_confidence = 0
    best_model = None
    for model, stats in model_stats.items():
        if stats['confidences']:
            avg_conf = sum(stats['confidences']) / len(stats['confidences'])
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_model = model
    
    for model, stats in model_stats.items():
        success_rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
        avg_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
        avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
        ocr_rate = stats['has_ocr'] / stats['successes'] if stats['successes'] > 0 else 0
        
        row_class = 'winner' if model == best_model else ''
        html_content += f"""
                <tr class="{row_class}">
                    <td>{model} {'👑' if model == best_model else ''}</td>
                    <td>{success_rate:.0%}</td>
                    <td>{avg_conf:.1%}</td>
                    <td>{avg_time:.1f}s</td>
                    <td>{ocr_rate:.0%}</td>
                </tr>
"""
    
    html_content += """
            </table>
        </div>
"""
    
    # Results for each image
    for image_path, results in all_results.items():
        image_name = Path(image_path).name
        
        html_content += f"""
        <div class="image-section">
            <h2>📸 Image: {image_name}</h2>
            <img src="file://{image_path}" class="image-preview" alt="{image_name}">
            
            <div class="comparison-grid">
"""
        
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
            model_class = model_classes.get(model_name, "")
            
            html_content += f"""
                <div class="model-card">
                    <div class="model-header {model_class}">
                        <span>{model_name}</span>
                        <span class="status-badge {'status-success' if success else 'status-failed'}">
                            {'✅ Success' if success else '❌ Failed'}
                        </span>
                    </div>
                    <div class="model-content">
"""
            
            if success:
                html_content += f"""
                        <div class="metric-row">
                            <span>Confidence:</span>
                            <span class="confidence {confidence_class}">{confidence:.1%}</span>
                        </div>
                        <div class="metric-row">
                            <span>Processing Time:</span>
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
    report_path = output_dir / "complete_vlm_comparison_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"📊 HTML report saved to: {report_path}")
    return report_path


def main():
    """Run complete VLM comparison on BMW document"""
    
    logger.info("Starting Complete VLM Comparison Test")
    logger.info("=" * 80)
    
    # Extract images from BMW PDF
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    image_dir = log_dir / "extracted_images"
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    # Extract images
    extracted_images = extract_images_from_pdf(pdf_path, image_dir, max_images=2)
    
    if not extracted_images:
        logger.error("No images extracted from PDF!")
        return
    
    logger.info(f"\n📸 Extracted {len(extracted_images)} images for testing")
    
    # Store all results
    all_results = {}
    
    # Test each image with all models
    for image_path in extracted_images:
        logger.info(f"\n{'='*80}")
        logger.info(f"🖼️  Testing image: {image_path.name}")
        logger.info('='*80)
        
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
        
        # Test 3: Pixtral
        logger.info("\n🚀 Starting Pixtral test...")
        logger.info("⏳ Note: Pixtral may take longer to load...")
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
        
        all_results[str(image_path)] = results
    
    # Save JSON results
    results_file = log_dir / "complete_vlm_comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 Results saved to: {results_file}")
    
    # Generate HTML report
    report_path = generate_comparison_html(all_results, log_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL COMPARISON SUMMARY")
    logger.info("="*80)
    
    # Calculate overall statistics
    model_totals = {}
    for image_path, results in all_results.items():
        for result in results:
            model = result['model']
            if model not in model_totals:
                model_totals[model] = {'success': 0, 'total': 0, 'confidence': [], 'time': []}
            
            model_totals[model]['total'] += 1
            if result.get('success'):
                model_totals[model]['success'] += 1
                model_totals[model]['confidence'].append(result.get('confidence', 0))
                model_totals[model]['time'].append(result.get('processing_time', 0))
    
    for model, stats in model_totals.items():
        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        avg_conf = sum(stats['confidence']) / len(stats['confidence']) if stats['confidence'] else 0
        avg_time = sum(stats['time']) / len(stats['time']) if stats['time'] else 0
        
        logger.info(f"\n{model}:")
        logger.info(f"   • Success Rate: {success_rate:.0f}% ({stats['success']}/{stats['total']})")
        if stats['confidence']:
            logger.info(f"   • Avg Confidence: {avg_conf:.1%}")
            logger.info(f"   • Avg Time: {avg_time:.1f}s")
    
    logger.info(f"\n📊 View detailed comparison at: {report_path}")
    logger.info(f"📁 JSON results at: {results_file}")
    logger.info(f"📄 Log file at: {log_dir}")
    logger.info("\n🎉 VLM comparison complete!")
    
    return {
        "log_dir": str(log_dir),
        "json_results": str(results_file),
        "html_report": str(report_path),
        "extracted_images": [str(p) for p in extracted_images]
    }


if __name__ == "__main__":
    paths = main()
    if paths:
        print(f"\n📍 Alle Ergebnisse verfügbar unter:")
        print(f"   • HTML Report: {paths['html_report']}")
        print(f"   • JSON Results: {paths['json_results']}")
        print(f"   • Log Directory: {paths['log_dir']}")