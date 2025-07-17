#!/usr/bin/env python3
"""
Standalone SmolDocling test using transformers library
No dependencies on project modules - everything self-contained
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from pdf2image import convert_from_path
import json
from datetime import datetime
import os

def test_smoldocling_transformers():
    """Test SmolDocling with transformers library directly"""
    
    print("=" * 80)
    print("🧪 SMOLDOCLING TRANSFORMERS STANDALONE TEST")
    print("=" * 80)
    
    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {DEVICE}")
    
    # PDF to test
    pdf_path = "data/input/Preview_BMW_3er_G20.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"📄 Testing with: {pdf_path}")
    
    # Convert PDF to images - using official 144 DPI
    print("\n📸 Converting PDF to images...")
    print("   Using DPI=144 (official SmolDocling recommendation)")
    
    try:
        images = convert_from_path(
            pdf_path,
            dpi=144,  # Official SmolDocling DPI
            first_page=1,
            last_page=1,  # Just test first page
            fmt='PNG'
        )
        
        if not images:
            print("❌ No images extracted from PDF")
            return
            
        page_image = images[0]
        print(f"✅ Image extracted: {page_image.size} (width x height)")
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return
    
    # Initialize processor and model
    print("\n🤖 Loading SmolDocling model with transformers...")
    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        
        # Load model with best practices
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            # Skip flash attention if not available
            # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        ).to(DEVICE)
        
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Processor type: {type(processor).__name__}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create input messages - exact format from HuggingFace docs
    print("\n📝 Creating input messages...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        },
    ]
    
    # Process with processor
    print("\n🔧 Processing input...")
    try:
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False)
        print(f"📝 Chat template applied: {text[:100]}...")
        
        # Process inputs
        inputs = processor(
            text=text,
            images=[page_image],
            return_tensors="pt"
        ).to(DEVICE)
        
        print(f"✅ Inputs prepared: {list(inputs.keys())}")
        
    except Exception as e:
        print(f"❌ Error processing inputs: {e}")
        return
    
    # Generate output
    print("\n🚀 Generating output...")
    start_time = datetime.now()
    
    try:
        with torch.no_grad():
            # Generate with different parameters to test
            generation_params = [
                {"max_new_tokens": 8192, "temperature": 0.0, "do_sample": False},
                {"max_new_tokens": 8192, "temperature": 0.1, "do_sample": True},
            ]
            
            for i, params in enumerate(generation_params):
                print(f"\n--- Test {i+1}: {params} ---")
                
                generated_ids = model.generate(
                    **inputs,
                    **params
                )
                
                # Decode output
                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Extract only the model's response
                if "Assistant:" in generated_text:
                    response = generated_text.split("Assistant:")[-1].strip()
                else:
                    response = generated_text
                
                print(f"\n🔍 RAW RESPONSE (first 1000 chars):")
                print("=" * 60)
                print(response[:1000])
                print("=" * 60)
                
                # Analyze response
                print(f"\n📊 Response Analysis:")
                print(f"   Total length: {len(response)} chars")
                print(f"   Lines: {len(response.split('\\n'))}")
                
                # Check for tags
                tags = {}
                for tag in ['<text>', '<other>', '<picture>', '<table>', '<formula>', '<code>', '<title>', '<section_header>']:
                    count = response.count(tag)
                    if count > 0:
                        tags[tag] = count
                
                print(f"   Tags found: {tags}")
                
                # Check for coordinate patterns
                import re
                coord_pattern = re.compile(r'(\d+)>(\d+)>(\d+)>(\d+)>')
                coords = coord_pattern.findall(response)
                print(f"   Coordinate patterns: {len(coords)}")
                
                # Save full response
                output_file = f"smoldocling_transformers_output_{i+1}_{datetime.now():%Y%m%d_%H%M%S}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Parameters: {params}\n")
                    f.write(f"Chat template: {text}\n")
                    f.write(f"Image size: {page_image.size}\n")
                    f.write(f"\n{'='*60}\nRAW RESPONSE:\n{'='*60}\n")
                    f.write(response)
                
                print(f"   💾 Full response saved to: {output_file}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"\n⏱️ Total processing time: {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error generating output: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_smoldocling_transformers()