#!/usr/bin/env python3
"""
Test SmolDocling using transformers library directly
Following the exact reference implementation from HuggingFace
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Configuration
MODEL_PATH = "ds4sd/SmolDocling-256M-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_transformers_direct():
    """Test with transformers library directly"""
    print("Testing SmolDocling with transformers library...")
    print(f"Device: {DEVICE}")
    
    # Create simple test image
    test_image = Image.new('RGB', (800, 600), 'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    
    # Add text with better spacing
    draw.text((50, 50), "Test Document", fill='black')
    draw.text((50, 100), "This is a simple test.", fill='black')
    draw.text((50, 150), "It should be easy to read.", fill='black')
    
    print("Created test image")
    
    # Load processor and model as per reference
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # Use eager attention since flash_attn is not installed
    ).to(DEVICE)
    
    print("Model loaded successfully")
    
    # Create messages in the correct format
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        }
    ]
    
    # Process the input
    print("Processing input...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[test_image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate output
    print("Generating output...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    
    # Print output
    print("\n" + "="*80)
    print("RAW OUTPUT:")
    print("="*80)
    print(generated_texts[0])
    print("="*80)
    
    # Try to extract just the assistant response
    if "Assistant:" in generated_texts[0]:
        assistant_response = generated_texts[0].split("Assistant:")[-1].strip()
        print("\nASSISTANT RESPONSE:")
        print("="*80)
        print(assistant_response)
        print("="*80)

if __name__ == "__main__":
    test_transformers_direct()