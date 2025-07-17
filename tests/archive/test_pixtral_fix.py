#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
import json
import base64
import time
import logging
from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_llava_client import TransformersLLaVAClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.parsers.interfaces.data_models import VisualElementType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test on first page only
test_image = Path("tests/debugging/key_pages_vlm_analysis/extracted_visuals/page_001_full.png")

if test_image.exists():
    logger.info("🚀 Running quick comparison test on BMW title page")
    
    with open(test_image, "rb") as f:
        image_data = f.read()
        base64_data = base64.b64encode(image_data).decode("utf-8")
    
    results = {
        "image_id": "page_001",
        "path": str(test_image),
        "page": 1,
        "description": "BMW 3 Series G20 - Title Page",
        "base64": base64_data,
        "analyses": []
    }
    
    # Initialize clients
    clients = {
        "Qwen2.5-VL-7B": TransformersQwen25VLClient(temperature=0.2, max_new_tokens=512),
        "LLaVA-1.6-Mistral-7B": TransformersLLaVAClient(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_8bit=True,
            temperature=0.2,
            max_new_tokens=512
        ),
        "Pixtral-12B": TransformersPixtralClient(
            temperature=0.3,
            max_new_tokens=512,
            load_in_8bit=True
        )
    }
    
    # Test each model
    for model_name, client in clients.items():
        logger.info(f"\n🤖 Testing {model_name}...")
        
        try:
            start = time.time()
            result = client.analyze_visual(
                image_data=image_data,
                analysis_focus="comprehensive"
            )
            elapsed = time.time() - start
            
            analysis = {
                "model": model_name,
                "success": result.success,
                "confidence": result.confidence,
                "description": result.description,
                "ocr_text": result.ocr_text,
                "extracted_data": result.extracted_data,
                "inference_time": elapsed,
                "error_message": result.error_message
            }
            
            if result.success:
                logger.info(f"✅ Success: {result.confidence:.0%} confidence")
                logger.info(f"📝 Description preview: {result.description[:200]}...")
            else:
                logger.info(f"❌ Failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            analysis = {
                "model": model_name,
                "success": False,
                "error_message": str(e),
                "confidence": 0,
                "inference_time": time.time() - start
            }
        
        results["analyses"].append(analysis)
        
        # Cleanup after each model
        if hasattr(client, "cleanup"):
            try:
                client.cleanup()
            except:
                pass
    
    # Save results
    output_file = Path("tests/debugging/pixtral_test_results.json")
    with open(output_file, "w") as f:
        json.dump([results], f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    
    # Generate simple HTML report
    from tests.create_html_from_results import generate_comparison_html
    html_path = Path("tests/debugging/pixtral_test_report.html")
    generate_comparison_html([results], html_path)
    logger.info(f"🌐 HTML report: {html_path}")
else:
    logger.error(f"Test image not found: {test_image}")