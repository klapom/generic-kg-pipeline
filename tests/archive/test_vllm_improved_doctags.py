#!/usr/bin/env python3
"""
Test vLLM SmolDocling with improved DocTags extraction
Integrating knowledge from transformers test
"""

import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image
from pdf2image import convert_from_path
import re

def test_vllm_improved():
    print("=" * 80)
    print("🧪 VLLM SMOLDOCLING TEST WITH IMPROVED DOCTAGS")
    print("=" * 80)
    
    # PDF to test
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Convert PDF to image (144 DPI)
    print("📸 Converting PDF to image...")
    images = convert_from_path(
        str(pdf_path),
        dpi=144,  # Official SmolDocling DPI
        first_page=1,
        last_page=1,
        fmt='PNG'
    )
    
    if not images:
        print("❌ No images extracted")
        return
        
    page_image = images[0]
    print(f"✅ Image size: {page_image.size}")
    
    # Initialize vLLM
    print("\n🤖 Initializing vLLM SmolDocling...")
    model = LLM(
        model="ds4sd/SmolDocling-256M-preview",
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.3,
        max_model_len=8192,
        trust_remote_code=False,
        dtype="auto"
    )
    print("✅ Model loaded")
    
    # Test different approaches
    test_configs = [
        {
            "name": "Standard (like current)",
            "template": "<|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:",
            "params": SamplingParams(temperature=0.0, max_tokens=8192)
        },
        {
            "name": "With generation prompt",
            "template": "<|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:",
            "params": SamplingParams(temperature=0.0, max_tokens=8192, skip_special_tokens=False)
        },
        {
            "name": "With stop tokens",
            "template": "<|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:",
            "params": SamplingParams(
                temperature=0.0, 
                max_tokens=8192,
                stop=["<end_of_utterance>", "</doctag>"],
                include_stop_str_in_output=True,
                skip_special_tokens=False
            )
        }
    ]
    
    for idx, config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Test {idx+1}: {config['name']}")
        print(f"{'='*80}")
        
        # Create multimodal input
        multimodal_input = {
            "prompt": config['template'],
            "multi_modal_data": {"image": page_image}
        }
        
        # Generate
        start_time = time.time()
        outputs = model.generate(multimodal_input, config['params'])
        gen_time = time.time() - start_time
        
        if outputs and outputs[0].outputs:
            output_text = outputs[0].outputs[0].text
            
            print(f"\n🔍 RAW OUTPUT (first 1000 chars):")
            print("-" * 60)
            print(output_text[:1000])
            print("-" * 60)
            
            # Analyze output
            print(f"\n📊 Analysis:")
            print(f"   Length: {len(output_text)} chars")
            print(f"   Generation time: {gen_time:.2f}s")
            
            # Check for tags
            tags = {}
            tag_patterns = [
                '<text>', '<other>', '<picture>', '<table>', '<formula>', 
                '<section_header', '<page_footer>', '<doctag>', '<loc_',
                '<title>', '<list_item>', '<code>'
            ]
            for tag in tag_patterns:
                count = output_text.count(tag)
                if count > 0:
                    tags[tag] = count
            print(f"   Tags found: {tags}")
            
            # Check coordinate formats
            bare_coords = len(re.findall(r'\d+>\d+>\d+>\d+>', output_text))
            loc_coords = len(re.findall(r'<loc_\d+>', output_text))
            print(f"   Bare coordinates (x>y>x2>y2>): {bare_coords}")
            print(f"   Loc tags (<loc_x>): {loc_coords}")
            
            # Save output
            output_file = f"vllm_improved_output_{idx+1}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Config: {config['name']}\n")
                f.write(f"Template: {config['template']}\n")
                f.write(f"Params: {config['params']}\n")
                f.write(f"\n{'='*60}\nRAW OUTPUT:\n{'='*60}\n")
                f.write(output_text)
            print(f"   💾 Saved to: {output_file}")
            
            # Try to extract the assistant response if present
            if "Assistant:" in output_text:
                response = output_text.split("Assistant:")[-1].strip()
                print(f"\n🤖 Extracted Assistant Response (first 500 chars):")
                print("-" * 40)
                print(response[:500])
                print("-" * 40)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_vllm_improved()