#!/usr/bin/env python3
"""Debug DocTags parsing with DoclingDocument"""

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Create a dummy image
img = Image.new('RGB', (500, 500), color='white')

# Test 1: Simple text
print("Test 1: Simple text DocTags")
doctags1 = '<paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph.</paragraph>'
print(f"DocTags: {doctags1}")

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags1], [img])
    doc = DoclingDocument(name='test1')
    doc.load_from_doctags(doctags_doc)
    
    print(f"Texts: {len(doc.texts)}")
    print(f"Text content: {doc.export_to_text()}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 2: With section header
print("Test 2: Section header and paragraph")
doctags2 = '<section_header><loc_10><loc_10><loc_490><loc_30>Test Section</section_header><paragraph><loc_10><loc_40><loc_490><loc_80>This is test content.</paragraph>'
print(f"DocTags: {doctags2}")

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags2], [img])
    doc = DoclingDocument(name='test2')
    doc.load_from_doctags(doctags_doc)
    
    print(f"Texts: {len(doc.texts)}")
    print(f"Text content: {doc.export_to_text()}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 3: With doctag wrapper (like SmolDocling output)
print("Test 3: With doctag wrapper")
doctags3 = '<doctag><section_header><loc_10><loc_10><loc_490><loc_30>Test Section</section_header><paragraph><loc_10><loc_40><loc_490><loc_80>This is test content.</paragraph></doctag>'
print(f"DocTags: {doctags3}")

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags3], [img])
    doc = DoclingDocument(name='test3')
    doc.load_from_doctags(doctags_doc)
    
    print(f"Texts: {len(doc.texts)}")
    print(f"Text content: {doc.export_to_text()}")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 4: Real SmolDocling output format
print("Test 4: Real SmolDocling output")
doctags4 = ' <doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>'
print(f"DocTags: {doctags4}")

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags4], [img])
    doc = DoclingDocument(name='test4')
    doc.load_from_doctags(doctags_doc)
    
    print(f"Texts: {len(doc.texts)}")
    print(f"Text content: {doc.export_to_text()}")
    
    # Check all properties
    print(f"Body children: {len(doc.body.children)}")
    print(f"Furniture children: {len(doc.furniture.children)}")
    print(f"Pages: {len(doc.pages)}")
    
    # Try to iterate all items
    items = list(doc.iterate_items())
    print(f"Total items: {len(items)}")
    for i, item in enumerate(items[:5]):
        print(f"  Item {i}: {type(item).__name__}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()