#!/usr/bin/env python3
"""Deep dive into why docling-core doesn't parse DocTags correctly"""

from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
import json

# Create a dummy image
img = Image.new('RGB', (500, 500), color='white')

print("=== DocTags Library Debug Analysis ===\n")

# Test cases with different DocTag formats
test_cases = [
    {
        "name": "Simple paragraph",
        "doctags": "<paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph.</paragraph>",
    },
    {
        "name": "With doctag wrapper",
        "doctags": "<doctag><paragraph><loc_10><loc_10><loc_490><loc_50>This is a test paragraph.</paragraph></doctag>",
    },
    {
        "name": "Multiple elements",
        "doctags": "<title><loc_10><loc_10><loc_490><loc_30>Test Title</title><paragraph><loc_10><loc_40><loc_490><loc_80>Test paragraph content.</paragraph>",
    },
    {
        "name": "Real SmolDocling format",
        "doctags": " <doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>",
    },
    {
        "name": "Complex real example",
        "doctags": '''<doctag><section_header><loc_50><loc_50><loc_450><loc_80>Technical Data</section_header>
<paragraph><loc_50><loc_90><loc_450><loc_120>Engine: 2.0L TwinPower Turbo</paragraph>
<table><loc_50><loc_130><loc_450><loc_200>Power|190 HP\nTorque|400 Nm</table>
</doctag>''',
    }
]

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"DocTags: {test['doctags'][:80]}...")
    print("-" * 40)
    
    try:
        # Create DocTagsDocument
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            [test['doctags']], 
            [img]
        )
        
        # Check DocTagsDocument structure
        print(f"DocTagsDocument type: {type(doctags_doc)}")
        print(f"DocTagsDocument attributes: {[attr for attr in dir(doctags_doc) if not attr.startswith('_')][:10]}...")
        
        # Try to access content
        if hasattr(doctags_doc, 'pages'):
            print(f"Pages: {len(doctags_doc.pages)}")
        if hasattr(doctags_doc, 'elements'):
            print(f"Elements: {len(doctags_doc.elements)}")
        if hasattr(doctags_doc, 'doctags'):
            print(f"Has doctags attribute")
            
        # Create DoclingDocument
        doc = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc,
            document_name="test"
        )
        
        print(f"\nDoclingDocument created:")
        print(f"  texts: {len(doc.texts)}")
        print(f"  tables: {len(doc.tables)}")
        print(f"  pictures: {len(doc.pictures)}")
        print(f"  body.children: {len(doc.body.children)}")
        print(f"  furniture.children: {len(doc.furniture.children)}")
        
        # Try to inspect the structure more
        if doc.texts:
            for i, text in enumerate(doc.texts[:3]):
                print(f"  Text[{i}]: type={type(text).__name__}, content={getattr(text, 'text', 'NO TEXT ATTR')[:50]}")
                
        if doc.body.children:
            for i, child in enumerate(doc.body.children[:3]):
                print(f"  Body child[{i}]: type={type(child).__name__}")
                attrs = [attr for attr in dir(child) if not attr.startswith('_')]
                print(f"    Attributes: {attrs[:10]}")
                if hasattr(child, 'text'):
                    print(f"    Text: {child.text[:50]}")
                if hasattr(child, 'content'):
                    print(f"    Content: {child.content[:50]}")
                    
        # Export to different formats to see what we get
        print(f"\nExport tests:")
        text_export = doc.export_to_text()
        print(f"  export_to_text(): '{text_export[:100]}'")
        
        md_export = doc.export_to_markdown()
        print(f"  export_to_markdown(): '{md_export[:100]}'")
        
        # Try to export to dict to see internal structure
        try:
            dict_export = doc.model_dump()
            print(f"  model_dump() keys: {list(dict_export.keys())}")
            if 'texts' in dict_export and dict_export['texts']:
                print(f"  First text item: {dict_export['texts'][0]}")
        except Exception as e:
            print(f"  model_dump() failed: {e}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Let's also check what methods are available on DocTagsDocument
print("\n\n=== DocTagsDocument Methods Analysis ===")
print("Methods that might help with parsing:")
methods = [m for m in dir(DocTagsDocument) if not m.startswith('_') and callable(getattr(DocTagsDocument, m))]
for method in sorted(methods):
    print(f"  - {method}")

# Check if there's a specific DocTags parser
print("\n\n=== Looking for DocTags Parser ===")
import docling_core
modules = [name for name in dir(docling_core) if 'tag' in name.lower() or 'pars' in name.lower()]
print(f"Relevant modules in docling_core: {modules}")

# Check submodules
try:
    import docling_core.types
    types_modules = [name for name in dir(docling_core.types) if not name.startswith('_')]
    print(f"Modules in docling_core.types: {types_modules}")
except Exception as e:
    print(f"Error exploring docling_core.types: {e}")