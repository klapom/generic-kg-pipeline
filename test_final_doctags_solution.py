#!/usr/bin/env python3
"""Test the final DocTags solution with transformation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient

# Create test client
client = VLLMSmolDoclingFinalClient(environment="testing")

# Test the transformation method
print("=== Testing DocTags Transformation ===\n")

test_cases = [
    {
        "name": "Simple paragraph",
        "input": "<paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph.</paragraph>",
    },
    {
        "name": "Section header",
        "input": "<section_header><loc_10><loc_10><loc_490><loc_30>Technical Data</section_header>",
    },
    {
        "name": "Complex SmolDocling output",
        "input": '''<doctag>
<page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>
<paragraph><loc_50><loc_50><loc_450><loc_100>BMW 3er Serie - Die ultimative Fahrmaschine.</paragraph>
<section_header><loc_50><loc_120><loc_450><loc_150>Technische Daten</section_header>
<paragraph><loc_50><loc_160><loc_450><loc_200>Motor: 2.0L TwinPower Turbo</paragraph>
<table><loc_50><loc_210><loc_450><loc_300>Leistung|190 PS\nDrehmoment|400 Nm</table>
</doctag>''',
    }
]

for test in test_cases:
    print(f"Test: {test['name']}")
    print(f"Input: {test['input'][:80]}...")
    
    # Transform
    transformed = client._transform_doctags(test['input'])
    print(f"Transformed: {transformed[:80]}...")
    
    # Test parsing
    try:
        img = Image.new('RGB', (500, 500), color='white')
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([transformed], [img])
        doc = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc,
            document_name="test"
        )
        
        print(f"✓ Parsed successfully!")
        print(f"  Texts: {len(doc.texts)}")
        print(f"  Tables: {len(doc.tables)}")
        print(f"  Pictures: {len(doc.pictures)}")
        
        # Show text content
        text_export = doc.export_to_text()
        if text_export.strip():
            print(f"  Text content:\n{text_export}")
        else:
            print(f"  No text content")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("-" * 80)

# Test complete workflow
print("\n=== Testing Complete Workflow ===")

# Simulate SmolDocling output
smoldocling_output = '''<doctag><picture><loc_0><loc_0><loc_500><loc_370></picture>
<paragraph><loc_10><loc_380><loc_490><loc_400>BMW - Freude am Fahren</paragraph>
</doctag>'''

print(f"SmolDocling output:\n{smoldocling_output}")

# Transform and parse
transformed = client._transform_doctags(smoldocling_output)
print(f"\nTransformed:\n{transformed}")

try:
    img = Image.new('RGB', (500, 500), color='white')
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([transformed], [img])
    doc = DoclingDocument.load_from_doctags(
        doctag_document=doctags_doc,
        document_name="bmw_page"
    )
    
    print(f"\n✓ Final result:")
    print(f"  Document name: {doc.name}")
    print(f"  Texts: {len(doc.texts)}")
    print(f"  Pictures: {len(doc.pictures)}")
    print(f"  Export to markdown:\n{doc.export_to_markdown()}")
    
except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()