#!/usr/bin/env python3
"""
Improved SmolDocling test using DoclingDocument conversion
Based on official code snippet with enhanced analysis
"""

import os
import torch
import time
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def process_document(image_path):
    timings = {}
    
    # Load image
    start_time = time.time()
    image = load_image(image_path)
    timings['image_loading'] = time.time() - start_time
    print(f"Image loading: {timings['image_loading']:.2f} seconds")
    print(f"  Image size: {image.size}")
    
    # Initialize processor and model
    start_time = time.time()
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    timings['processor_init'] = time.time() - start_time
    print(f"Processor initialization: {timings['processor_init']:.2f} seconds")
    
    start_time = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
        # _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",  # Skip if not available
    ).to(DEVICE)
    timings['model_init'] = time.time() - start_time
    print(f"Model initialization: {timings['model_init']:.2f} seconds")
    
    # Create input messages
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this page to docling."}
        ]
    }]
    
    # Process image and prepare inputs
    start_time = time.time()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(f"\nApplied chat template: {prompt[:100]}...")
    
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    timings['input_processing'] = time.time() - start_time
    print(f"Input processing: {timings['input_processing']:.2f} seconds")
    
    # Generate outputs
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    timings['model_inference'] = time.time() - start_time
    print(f"Model inference: {timings['model_inference']:.2f} seconds")
    
    # Decode outputs
    start_time = time.time()
    trimmed_generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    timings['output_decoding'] = time.time() - start_time
    print(f"Output decoding: {timings['output_decoding']:.2f} seconds")
    
    # Create document
    start_time = time.time()
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        timings['document_creation'] = time.time() - start_time
        print(f"Document creation: {timings['document_creation']:.2f} seconds")
        document_created = True
    except Exception as e:
        print(f"⚠️ Document creation failed: {e}")
        doc = None
        document_created = False
        timings['document_creation'] = time.time() - start_time
    
    # Calculate total time
    timings['total'] = sum(timings.values())
    print(f"Total processing time: {timings['total']:.2f} seconds")
    
    return doc, doctags, document_created

def analyze_doctags(doctags):
    """Analyze the raw DocTags output"""
    print("\n" + "="*80)
    print("📊 DOCTAGS ANALYSIS:")
    print("="*80)
    
    # Basic stats
    print(f"Total length: {len(doctags)} chars")
    print(f"Total lines: {len(doctags.split(chr(10)))}")
    
    # Count tags
    import re
    tags = {}
    for tag in ['<text>', '<other>', '<picture>', '<table>', '<formula>', '<code>', 
                '<title>', '<section_header>', '<list_item>', '<otsl>', '<logo>']:
        count = doctags.count(tag)
        if count > 0:
            tags[tag] = count
    
    print(f"Tags found: {tags}")
    
    # Count coordinate patterns
    coord_pattern = re.compile(r'(\d+)>(\d+)>(\d+)>(\d+)>')
    coords = coord_pattern.findall(doctags)
    print(f"Coordinate patterns: {len(coords)}")
    
    # Show first 1000 chars
    print(f"\nFirst 1000 chars of DocTags:")
    print("-"*60)
    print(doctags[:1000])
    print("-"*60)

def main():
    print("="*80)
    print("🧪 IMPROVED SMOLDOCLING TEST WITH DOCLING CONVERSION")
    print("="*80)
    print(f"Using device: {DEVICE}")
    
    # Test with extracted images
    image_dir = Path("tests/debugging/complete_workflow/extracted_images")
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        print("Looking for alternatives...")
        
        # Try to find a single PNG image first
        alt_images = list(Path(".").glob("**/*.png"))[:5]
        if alt_images:
            print(f"Found {len(alt_images)} PNG images to test with")
            test_images = alt_images[:2]  # Test first 2
        else:
            print("No images found to test")
            return
    else:
        # Use extracted images
        test_images = sorted(image_dir.glob("page_*.png"))[:3]  # Test first 3 pages
        print(f"Found {len(test_images)} extracted page images")
    
    # Process each image
    all_results = []
    
    for idx, image_path in enumerate(test_images):
        print(f"\n{'='*80}")
        print(f"📄 Processing image {idx+1}/{len(test_images)}: {image_path.name}")
        print(f"{'='*80}")
        
        # Record total execution time for this image
        total_start_time = time.time()
        
        # Process document
        doc, doctags, doc_created = process_document(str(image_path))
        
        # Analyze DocTags
        analyze_doctags(doctags)
        
        # Save raw DocTags
        output_file = f"doctags_output_{image_path.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(doctags)
        print(f"\n💾 Raw DocTags saved to: {output_file}")
        
        # Print markdown if document was created
        if doc_created and doc:
            print("\n📝 Markdown Export:")
            print("-"*60)
            markdown = doc.export_to_markdown()
            print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
            
            # Save markdown
            md_file = f"markdown_output_{image_path.stem}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"\n💾 Markdown saved to: {md_file}")
        
        print(f"\nTotal execution time for this image: {time.time() - total_start_time:.2f} seconds")
        
        all_results.append({
            'image': image_path.name,
            'doctags_length': len(doctags),
            'doc_created': doc_created,
            'markdown_length': len(doc.export_to_markdown()) if doc_created and doc else 0
        })
    
    # Summary
    print("\n" + "="*80)
    print("📊 PROCESSING SUMMARY:")
    print("="*80)
    for result in all_results:
        print(f"Image: {result['image']}")
        print(f"  DocTags length: {result['doctags_length']} chars")
        print(f"  Document created: {'✅' if result['doc_created'] else '❌'}")
        if result['doc_created']:
            print(f"  Markdown length: {result['markdown_length']} chars")
        print()

if __name__ == "__main__":
    main()