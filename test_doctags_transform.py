#!/usr/bin/env python3
"""Test transforming DocTags to make them compatible with docling-core"""

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
import re

def transform_doctags(doctags: str) -> str:
    """Transform SmolDocling DocTags to docling-core compatible format"""
    
    # Mapping of SmolDocling tags to docling-core compatible tags
    tag_mapping = {
        'paragraph': 'text',
        'section_header': 'text',  # or 'title' depending on context
        'page_header': 'text',
        'page_footer': 'text',
        # Keep these as-is
        'title': 'title',
        'caption': 'caption',
        'formula': 'formula',
        'code': 'code',
        'table': 'table',
        'picture': 'picture',
        'figure': 'picture',
    }
    
    # Transform tags
    transformed = doctags
    for old_tag, new_tag in tag_mapping.items():
        # Replace opening tags
        transformed = re.sub(f'<{old_tag}>', f'<{new_tag}>', transformed)
        # Replace closing tags
        transformed = re.sub(f'</{old_tag}>', f'</{new_tag}>', transformed)
    
    return transformed

# Test the transformation
img = Image.new('RGB', (500, 500), color='white')

test_cases = [
    {
        "name": "Paragraph to text",
        "original": "<paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph.</paragraph>",
    },
    {
        "name": "Section header",
        "original": "<section_header><loc_10><loc_10><loc_490><loc_30>Technical Data</section_header>",
    },
    {
        "name": "Complex example",
        "original": '''<doctag>
<section_header><loc_50><loc_50><loc_450><loc_80>Technical Data</section_header>
<paragraph><loc_50><loc_90><loc_450><loc_120>Engine: 2.0L TwinPower Turbo</paragraph>
<table><loc_50><loc_130><loc_450><loc_200>Power|190 HP\nTorque|400 Nm</table>
</doctag>''',
    }
]

print("=== Testing DocTags Transformation ===\n")

for test in test_cases:
    print(f"Test: {test['name']}")
    print(f"Original: {test['original'][:80]}...")
    
    # Transform
    transformed = transform_doctags(test['original'])
    print(f"Transformed: {transformed[:80]}...")
    
    try:
        # Parse original
        doctags_doc_orig = DocTagsDocument.from_doctags_and_image_pairs([test['original']], [img])
        doc_orig = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc_orig,
            document_name="original"
        )
        
        # Parse transformed
        doctags_doc_trans = DocTagsDocument.from_doctags_and_image_pairs([transformed], [img])
        doc_trans = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc_trans,
            document_name="transformed"
        )
        
        print(f"\nResults:")
        print(f"  Original - texts: {len(doc_orig.texts)}, export: '{doc_orig.export_to_text()}'")
        print(f"  Transformed - texts: {len(doc_trans.texts)}, export: '{doc_trans.export_to_text()}'")
        
        if doc_trans.texts:
            for i, text in enumerate(doc_trans.texts):
                print(f"    Text[{i}]: '{text.text}'")
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("-" * 80)

# Test with real SmolDocling output
print("\n=== Real SmolDocling Example ===")
real_doctags = '''<doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>
<paragraph><loc_50><loc_50><loc_450><loc_100>BMW 3er Serie - Die ultimative Fahrmaschine.</paragraph>
<section_header><loc_50><loc_120><loc_450><loc_150>Technische Daten</section_header>
<paragraph><loc_50><loc_160><loc_450><loc_200>Motor: 2.0L TwinPower Turbo</paragraph>
</doctag>'''

print(f"Original SmolDocling output:")
print(real_doctags)

transformed = transform_doctags(real_doctags)
print(f"\nTransformed:")
print(transformed)

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([transformed], [img])
    doc = DoclingDocument.load_from_doctags(
        doctag_document=doctags_doc,
        document_name="bmw"
    )
    
    print(f"\nParsed successfully!")
    print(f"Texts: {len(doc.texts)}")
    print(f"Export to text:\n{doc.export_to_text()}")
    
except Exception as e:
    print(f"ERROR: {e}")