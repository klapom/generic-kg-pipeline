#!/usr/bin/env python3
"""Check which tags are supported by docling-core"""

from docling_core.types.doc.labels import DocItemLabel, GroupLabel
from docling_core.types.doc.document import DoclingDocument

print("=== Supported DocItemLabel values ===")
for label in DocItemLabel:
    print(f"  {label.value}: {label.name}")

print("\n=== Supported GroupLabel values ===")
for label in GroupLabel:
    print(f"  {label.value}: {label.name}")

# Let's check if there's a mapping between DocTags and DocItemLabels
print("\n=== Testing different tag names ===")

from PIL import Image
from docling_core.types.doc.document import DocTagsDocument

img = Image.new('RGB', (500, 500), color='white')

# Test different tag names that might work
test_tags = [
    "paragraph",
    "para",
    "p",
    "text",
    "textblock",
    "text_block",
    "section_header", 
    "section-header",
    "sectionheader",
    "heading",
    "header",
    "h1", "h2", "h3",
    "table",
    "list",
    "list_item",
    "caption",
    "figure",
    "formula",
    "code",
    "code_block"
]

working_tags = []
not_working_tags = []

for tag in test_tags:
    doctags = f"<{tag}><loc_10><loc_10><loc_490><loc_50>Test content</{tag}>"
    
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [img])
        doc = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc,
            document_name="test"
        )
        
        has_content = (len(doc.texts) > 0 or len(doc.tables) > 0 or 
                      len(doc.pictures) > 0 or len(doc.body.children) > 0)
        
        if has_content:
            working_tags.append(tag)
            # Check what type it became
            if doc.texts:
                print(f"✓ '{tag}' -> texts[0].label = {doc.texts[0].label}")
            elif doc.tables:
                print(f"✓ '{tag}' -> table")
            elif doc.pictures:
                print(f"✓ '{tag}' -> picture")
        else:
            not_working_tags.append(tag)
            
    except Exception as e:
        not_working_tags.append(f"{tag} (error: {e})")

print(f"\n✓ Working tags: {working_tags}")
print(f"✗ Not working tags: {not_working_tags}")

# Let's also check the source code to see how DocTags are parsed
print("\n=== Checking DocTagsDocument source ===")
import inspect
try:
    source = inspect.getsource(DocTagsDocument.from_doctags_and_image_pairs)
    print("Source code preview:")
    print(source[:500] + "...")
except Exception as e:
    print(f"Could not get source: {e}")

# Check if there's a tag mapping somewhere
print("\n=== Looking for tag mappings ===")
import docling_core.types.doc.labels as labels_module
mappings = [attr for attr in dir(labels_module) if 'map' in attr.lower() or 'tag' in attr.lower()]
print(f"Potential mappings in labels module: {mappings}")