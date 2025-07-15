#!/usr/bin/env python3
"""
Debug vLLM SmolDocling output issues
"""

import os
from vllm import LLM, SamplingParams
from PIL import Image
import torch

# Ensure we're using the right settings
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow downloading if needed
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'  # Try different backend

# Configuration
MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
PROMPT_TEXT = "Convert this page to docling."

def test_vllm_debug():
    """Test vLLM with debugging"""
    print("Testing vLLM SmolDocling with debugging...")
    
    # Create simple test image
    test_image = Image.new('RGB', (800, 600), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    
    # Add text
    draw.text((50, 50), "Test Document", fill='black')
    draw.text((50, 100), "This is a simple test.", fill='black')
    draw.text((50, 150), "It should be easy to read.", fill='black')
    
    print("Created test image")
    
    # Initialize LLM with specific settings
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.3,
        max_model_len=8192,
        trust_remote_code=False,  # Try without remote code
        dtype="bfloat16"  # Match transformers dtype
    )
    
    # Try different sampling parameters
    sampling_configs = [
        # Reference config
        {"temperature": 0.0, "max_tokens": 8192},
        # Try with repetition penalty
        {"temperature": 0.0, "max_tokens": 8192, "repetition_penalty": 1.1},
        # Try with lower temperature
        {"temperature": 0.1, "max_tokens": 8192, "top_p": 0.95},
    ]
    
    for i, params in enumerate(sampling_configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {params}")
        print('='*80)
        
        sampling_params = SamplingParams(**params)
        
        # Try different prompt formats
        prompts = [
            # Exact reference format
            f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance> Assistant:",
            # Without the space before Assistant
            f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>Assistant:",
            # Try with newline
            f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>\nAssistant:"
        ]
        
        for j, chat_template in enumerate(prompts):
            print(f"\nPrompt variant {j+1}: {repr(chat_template[:50])+'...'}")
            
            # Create multimodal input
            multimodal_input = {
                "prompt": chat_template,
                "multi_modal_data": {"image": test_image}
            }
            
            # Generate
            outputs = llm.generate(multimodal_input, sampling_params)
            
            if outputs and outputs[0].outputs:
                output_text = outputs[0].outputs[0].text
                print(f"Output length: {len(output_text)} chars")
                print(f"First 200 chars: {repr(output_text[:200])}")
                
                # Check if output looks like DocTags
                if "<text>" in output_text[:100]:
                    print("✓ Output contains DocTags!")
                else:
                    print("✗ Output doesn't look like DocTags")

if __name__ == "__main__":
    test_vllm_debug()