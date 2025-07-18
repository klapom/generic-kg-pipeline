#!/usr/bin/env python3
"""Test if we can hack/extend the tag mapping in docling-core"""

from PIL import Image
from docling_core.types.doc import DoclingDocument, DocItemLabel
from docling_core.types.doc.document import DocTagsDocument
import docling_core.types.doc.document as doc_module

# Get the original load_from_doctags method
original_load_from_doctags = DoclingDocument.load_from_doctags

print("=== Original tag mapping ===")
# Let's see if we can access the tag mapping
import inspect
source = inspect.getsource(original_load_from_doctags)

# Extract tag_to_doclabel mapping from source
import re
mapping_match = re.search(r'tag_to_doclabel = \{([^}]+)\}', source, re.DOTALL)
if mapping_match:
    print("Found mapping:")
    mapping_text = mapping_match.group(1)
    for line in mapping_text.strip().split('\n'):
        if ':' in line:
            print(f"  {line.strip()}")

# Let's try to monkey-patch the method
def patched_load_from_doctags(doctag_document: DocTagsDocument, document_name: str = "Document") -> "DoclingDocument":
    """Patched version that handles paragraph tags"""
    
    # First, let's transform the doctags in the document
    for page in doctag_document.pages:
        if hasattr(page, 'tokens') and page.tokens:
            # Replace paragraph with text
            page.tokens = page.tokens.replace('<paragraph>', '<text>')
            page.tokens = page.tokens.replace('</paragraph>', '</text>')
            # Replace section_header with section_header_level_1
            page.tokens = page.tokens.replace('<section_header>', '<section_header_level_1>')
            page.tokens = page.tokens.replace('</section_header>', '</section_header_level_1>')
    
    # Now call the original method
    return original_load_from_doctags(doctag_document, document_name)

# Replace the method
DoclingDocument.load_from_doctags = staticmethod(patched_load_from_doctags)

# Test it
print("\n=== Testing patched version ===")

img = Image.new('RGB', (500, 500), color='white')

test_doctags = '''<doctag>
<paragraph><loc_50><loc_50><loc_450><loc_100>This is a paragraph that should now work.</paragraph>
<section_header><loc_50><loc_120><loc_450><loc_150>Section Header Test</section_header>
<title><loc_50><loc_170><loc_450><loc_200>Title Test</title>
</doctag>'''

print(f"Test DocTags:\n{test_doctags}")

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([test_doctags], [img])
    doc = DoclingDocument.load_from_doctags(
        doctag_document=doctags_doc,
        document_name="test"
    )
    
    print(f"\nResults:")
    print(f"  Texts: {len(doc.texts)}")
    print(f"  Export to text:\n{doc.export_to_text()}")
    
    for i, text in enumerate(doc.texts):
        print(f"  Text[{i}]: label={text.label}, content='{text.text}'")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test with real BMW example
print("\n=== Real BMW Example ===")
bmw_doctags = '''<doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>
<paragraph><loc_50><loc_50><loc_450><loc_100>BMW 3er Serie - Die ultimative Fahrmaschine.</paragraph>
<section_header><loc_50><loc_120><loc_450><loc_150>Technische Daten</section_header>
<paragraph><loc_50><loc_160><loc_450><loc_200>Motor: 2.0L TwinPower Turbo</paragraph>
</doctag>'''

try:
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([bmw_doctags], [img])
    doc = DoclingDocument.load_from_doctags(
        doctag_document=doctags_doc,
        document_name="bmw"
    )
    
    print(f"BMW Results:")
    print(f"  Texts: {len(doc.texts)}")
    print(f"  Export to markdown:\n{doc.export_to_markdown()}")
    
except Exception as e:
    print(f"ERROR: {e}")