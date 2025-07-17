#!/usr/bin/env python3
"""Test Qwen2.5-VL with vLLM directly using updated transformers"""

import asyncio
from pathlib import Path
import json
import logging
from datetime import datetime
from PIL import Image

# Setup logging
log_dir = Path("tests/debugging/vllm_qwen25vl_direct")
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

async def test_vllm_direct():
    """Test Qwen2.5-VL with vLLM directly"""
    
    try:
        # Import vllm
        logger.info("Importing vLLM...")
        from vllm import LLM, SamplingParams
        from vllm.multimodal import MultiModalDataBuiltins
        
        # Initialize model
        logger.info("Initializing Qwen2.5-VL-7B-Instruct with vLLM...")
        llm = LLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.7,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 5}
        )
        
        logger.info("Model loaded successfully!")
        
        # Test image
        test_image_path = Path("tests/debugging/visual_elements_with_vlm/extracted_images/visual_page3_element2.png")
        
        if not test_image_path.exists():
            logger.error(f"Test image not found: {test_image_path}")
            return
        
        # Load image
        image = Image.open(test_image_path)
        
        # Create prompt with image placeholder
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in detail. What do you see?<|im_end|>\n<|im_start|>assistant\n"
        
        # Alternative prompt format for Qwen2.5-VL
        prompt_alt = "Describe this image in detail. What do you see?"
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=512,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # Try different approaches
        logger.info("Testing vLLM inference...")
        
        # Approach 1: Direct image input
        try:
            logger.info("Approach 1: Direct image input")
            outputs = llm.generate(
                {
                    "prompt": prompt_alt,
                    "multi_modal_data": {"image": image}
                },
                sampling_params=sampling_params
            )
            
            if outputs:
                result = outputs[0].outputs[0].text
                logger.info(f"Success! Response: {result[:200]}...")
                
                # Save result
                save_result({
                    "approach": "direct_image",
                    "success": True,
                    "response": result
                }, log_dir)
                
        except Exception as e:
            logger.error(f"Approach 1 failed: {e}")
        
        # Approach 2: Using processor
        try:
            logger.info("Approach 2: Using processor")
            from transformers import AutoProcessor
            
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            
            # Process image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            logger.info(f"Processed prompt: {text[:100]}...")
            
            outputs = llm.generate(
                text,
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            if outputs:
                result = outputs[0].outputs[0].text
                logger.info(f"Success! Response: {result[:200]}...")
                
                # Save result
                save_result({
                    "approach": "processor",
                    "success": True,
                    "response": result,
                    "prompt": text
                }, log_dir)
                
        except Exception as e:
            logger.error(f"Approach 2 failed: {e}")
        
        logger.info("Test completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

def save_result(result, output_dir):
    """Save test result"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{result['approach']}_{timestamp}.json"
    
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Result saved to {filename}")

if __name__ == "__main__":
    asyncio.run(test_vllm_direct())