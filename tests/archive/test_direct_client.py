#!/usr/bin/env python3
"""
Test SmolDocling client directly without model manager
"""

import os
from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image

# Set environment
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

def test_direct_client():
    """Test direct client initialization"""
    print("Testing direct SmolDocling client...")
    
    # Create test PDF if needed
    pdf_path = Path("data/input/test_simple.pdf")
    if not pdf_path.exists():
        print("Creating test PDF...")
        from create_test_pdf import create_test_pdf
        create_test_pdf()
    
    # Initialize SmolDocling model directly
    print("Initializing vLLM model directly...")
    model = LLM(
        model="ds4sd/SmolDocling-256M-preview",
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.12,
        max_model_len=8192,
        trust_remote_code=False,
        dtype="bfloat16"
    )
    
    # Convert PDF to image
    from pdf2image import convert_from_path
    print("Converting PDF to image...")
    images = convert_from_path(str(pdf_path), dpi=300, fmt='PNG')
    
    if not images:
        print("No images extracted from PDF")
        return
    
    test_image = images[0]
    
    # Resize if needed
    max_size = 1536
    if max(test_image.size) > max_size:
        test_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Test with exact prompt format
    PROMPT_TEXT = "Convert this page to docling."
    chat_template = f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance> Assistant:"
    
    # Create multimodal input
    multimodal_input = {
        "prompt": chat_template,
        "multi_modal_data": {"image": test_image}
    }
    
    # Generate with different sampling params
    configs = [
        {"temperature": 0.0, "max_tokens": 8192},
        {"temperature": 0.0, "max_tokens": 8192, "repetition_penalty": 1.5},
    ]
    
    for i, params in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Test {i+1}: {params}")
        print('='*80)
        
        sampling_params = SamplingParams(**params)
        outputs = model.generate(multimodal_input, sampling_params)
        
        if outputs and outputs[0].outputs:
            output_text = outputs[0].outputs[0].text
            print(f"Output length: {len(output_text)} chars")
            print(f"First 500 chars:\n{output_text[:500]}")
            
            # Check quality
            if "<text>" in output_text and "<loc_" in output_text:
                print("✓ Output looks like proper DocTags!")
            else:
                print("✗ Output quality issues detected")

if __name__ == "__main__":
    test_direct_client()