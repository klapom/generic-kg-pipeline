#!/usr/bin/env python3
"""Test Qwen2.5-VL using Transformers directly (fallback approach)"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime
import torch
from PIL import Image
from io import BytesIO

# Setup logging
log_dir = Path("tests/debugging/qwen25vl_transformers")
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

async def test_transformers_approach():
    """Test Qwen2.5-VL with Transformers"""
    
    try:
        # Import transformers components
        logger.info("Importing Qwen2.5-VL from transformers...")
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        # Load model and processor
        logger.info("Loading Qwen2.5-VL-7B-Instruct model...")
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        logger.info("Model loaded successfully!")
        
        # Test images
        test_images = [
            {
                "path": Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png"),
                "name": "BMW front view with annotations"
            }
        ]
        
        results = []
        
        for img_info in test_images:
            if not img_info["path"].exists():
                logger.warning(f"Image not found: {img_info['path']}")
                continue
            
            logger.info(f"\nTesting: {img_info['name']}")
            
            try:
                # Load image
                image = Image.open(img_info["path"])
                
                # Prepare messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in detail. What do you see?"
                            }
                        ]
                    }
                ]
                
                # Process with model
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                logger.info(f"Image inputs: {len(image_inputs)}, Video inputs: {len(video_inputs)}")
                
                inputs = processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                
                # Generate
                logger.info("Generating response...")
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                logger.info(f"Response: {output_text[:200]}...")
                
                results.append({
                    "image": img_info["name"],
                    "success": True,
                    "response": output_text
                })
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
                results.append({
                    "image": img_info["name"],
                    "success": False,
                    "error": str(e)
                })
        
        # Save results
        results_file = log_dir / "transformers_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

def process_vision_info(messages):
    """Extract images and videos from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "image":
                    image_inputs.append(item["image"])
                elif item["type"] == "video":
                    video_inputs.append(item["video"])
    
    return image_inputs, video_inputs

if __name__ == "__main__":
    asyncio.run(test_transformers_approach())