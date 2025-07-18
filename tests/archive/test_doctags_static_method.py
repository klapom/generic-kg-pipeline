#!/usr/bin/env python3
"""Test the corrected static method usage for DoclingDocument"""

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Create a dummy image
img = Image.new('RGB', (500, 500), color='white')

# Test DocTags from SmolDocling output
doctags_examples = [
    # Example 1: Picture only (like BMW page 1)
    ('<doctag><picture><loc_0><loc_0><loc_500><loc_370><other></picture>', "Picture only"),
    
    # Example 2: Page header
    ('<doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>', "Page header"),
    
    # Example 3: Text content
    ('<doctag><paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph with actual text content.</paragraph></doctag>', "Paragraph"),
    
    # Example 4: Multiple elements
    ('<doctag><title><loc_10><loc_10><loc_490><loc_30>Test Title</title><paragraph><loc_10><loc_40><loc_490><loc_80>Test content here.</paragraph></doctag>', "Title and paragraph"),
]

print("Testing static method fix for DoclingDocument.load_from_doctags()\n")

for doctags, description in doctags_examples:
    print(f"Test: {description}")
    print(f"DocTags: {doctags[:80]}...")
    
    try:
        # Create DocTagsDocument
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [img])
        
        # OLD WAY (WRONG - leaves doc empty):
        # doc = DoclingDocument(name="test")
        # doc.load_from_doctags(doctags_doc)
        
        # NEW WAY (CORRECT - returns populated document):
        doc = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc,
            document_name="test"
        )
        
        print(f"✓ Success: texts={len(doc.texts)}, tables={len(doc.tables)}, pictures={len(doc.pictures)}")
        
        # Try to get text
        text_content = doc.export_to_text()
        if text_content.strip():
            print(f"  Text content: {text_content.strip()[:100]}...")
        else:
            print(f"  No text content extracted")
            
        # Check body and furniture
        print(f"  Body children: {len(doc.body.children)}")
        print(f"  Furniture children: {len(doc.furniture.children)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 80)