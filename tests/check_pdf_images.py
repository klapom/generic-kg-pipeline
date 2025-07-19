#!/usr/bin/env python3
"""
Check if PDF contains embedded images using PyMuPDF
"""

import fitz  # PyMuPDF
from pathlib import Path

# Find BMW PDFs
pdf_files = list(Path("data/input").glob("*BMW*.pdf")) + list(Path("data/input").glob("*bmw*.pdf"))

for pdf_path in pdf_files[:3]:  # Check first 3 files
    print(f"\nChecking: {pdf_path.name}")
    print("=" * 50)
    
    doc = fitz.open(str(pdf_path))
    
    total_images = 0
    
    for page_num in range(min(10, len(doc))):  # Check first 10 pages
        page = doc[page_num]
        
        # Get image list
        image_list = page.get_images(full=True)
        
        if image_list:
            print(f"\nPage {page_num + 1}: Found {len(image_list)} images")
            total_images += len(image_list)
            
            for img_index, img in enumerate(image_list[:3]):  # Show first 3 images
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                print(f"  Image {img_index + 1}:")
                print(f"    - Size: {pix.width}x{pix.height}")
                print(f"    - Colorspace: {pix.colorspace.name if pix.colorspace else 'Unknown'}")
                print(f"    - Alpha: {pix.alpha}")
                pix = None  # Free memory
    
    print(f"\nTotal embedded images in first 10 pages: {total_images}")
    
    # Also check for any drawing objects that might be vector graphics
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        drawings = page.get_drawings()
        if drawings:
            print(f"\nPage {page_num + 1}: Found {len(drawings)} vector drawings")
    
    doc.close()