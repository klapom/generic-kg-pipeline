#!/usr/bin/env python3
"""
Direct vLLM test following exact reference implementation
"""

import time
import os
from vllm import LLM, SamplingParams
from PIL import Image
from pathlib import Path

# Configuration - EXACT from reference
MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
PROMPT_TEXT = "Convert page to Docling."

def test_direct_vllm():
    """Test with exact reference implementation"""
    print("Testing direct vLLM implementation...")
    
    # Create simple test image
    test_image = Image.new('RGB', (800, 600), 'white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_image)
    
    # Add text
    draw.text((50, 50), "Test Document", fill='black')
    draw.text((50, 100), "This is a simple test.", fill='black')
    draw.text((50, 150), "It should be easy to read.", fill='black')
    
    print("Created test image")
    
    # Initialize LLM - EXACT from reference with reduced GPU memory
    llm = LLM(model=MODEL_PATH, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.3)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192)
    
    # EXACT chat template from reference
    chat_template = f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance> Assistant:"
    
    print("Initialized model")
    
    # Create multimodal input
    multimodal_input = {
        "prompt": chat_template,
        "multi_modal_data": {"image": test_image}
    }
    
    print("Processing...")
    start_time = time.time()
    
    # Generate
    outputs = llm.generate(multimodal_input, sampling_params)
    
    processing_time = time.time() - start_time
    print(f"Processing took {processing_time:.2f} seconds")
    
    # Print output
    if outputs:
        output_text = outputs[0].outputs[0].text if outputs[0].outputs else "No output"
        print("\n" + "="*80)
        print("RAW OUTPUT:")
        print("="*80)
        print(output_text)
        print("="*80)
        print(f"Output length: {len(output_text)} chars")
    else:
        print("No outputs generated")

if __name__ == "__main__":
    test_direct_vllm()