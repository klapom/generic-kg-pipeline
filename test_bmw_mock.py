#!/usr/bin/env python3
"""Mock test for BMW document processing without GPU"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
import logging
from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample DocTags outputs from SmolDocling
SAMPLE_DOCTAGS = [
    # Page 1 - mostly visual
    '<doctag><picture><loc_0><loc_0><loc_500><loc_370><other></picture>',
    
    # Page 2 - with text
    '''<doctag><page_header><loc_29><loc_11><loc_32><loc_17>2</page_header>
<paragraph><loc_50><loc_50><loc_450><loc_100>BMW 3er Serie - Die ultimative Fahrmaschine.</paragraph>
<section_header><loc_50><loc_120><loc_450><loc_150>Technische Daten</section_header>
<paragraph><loc_50><loc_160><loc_450><loc_200>Motor: 2.0L TwinPower Turbo</paragraph>
<table><loc_50><loc_210><loc_450><loc_300>Leistung|190 PS\nDrehmoment|400 Nm\nBeschleunigung|7.1s</table>
</doctag>''',
    
    # Page 3 - mixed content
    '''<doctag><title><loc_50><loc_30><loc_450><loc_60>Ausstattung</title>
<paragraph><loc_50><loc_70><loc_450><loc_120>Die neue BMW 3er Serie bietet modernste Technologie.</paragraph>
<picture><loc_50><loc_130><loc_450><loc_250>interior.jpg</picture>
<caption><loc_50><loc_260><loc_450><loc_280>Hochwertiges Interieur mit Ambient Beleuchtung</caption>
</doctag>'''
]

def test_doctags_processing():
    """Test DocTags processing with mock data"""
    
    print("Testing DocTags processing with improved fallback...\n")
    
    # Create mock client
    client = VLLMSmolDoclingFinalClient(environment="testing")
    
    # Mock the vLLM model call
    def mock_generate_doctags(page_image):
        # Return sample DocTags based on page number
        page_num = getattr(mock_generate_doctags, 'call_count', 0)
        mock_generate_doctags.call_count = page_num + 1
        return SAMPLE_DOCTAGS[min(page_num, len(SAMPLE_DOCTAGS)-1)]
    
    mock_generate_doctags.call_count = 0
    
    # Patch the method
    with patch.object(client, '_generate_doctags', side_effect=mock_generate_doctags):
        # Create dummy images
        page_images = [Image.new('RGB', (500, 500), color='white') for _ in range(3)]
        
        # Process each page
        for page_num, page_image in enumerate(page_images, 1):
            print(f"Processing page {page_num}...")
            
            doctags = client._generate_doctags(page_image)
            print(f"DocTags preview: {doctags[:100]}...")
            
            # Create a mock PDF doc
            class MockPdfDoc:
                def __getitem__(self, index):
                    return Mock(rect=Mock(width=500, height=500))
            pdf_doc = MockPdfDoc()
            
            try:
                # Test the direct parsing method
                page = client._parse_doctags_directly(doctags, page_num, pdf_doc, page_image)
                
                print(f"✓ Page {page_num} results:")
                print(f"  Text: {page.text[:100] if page.text and page.text != '[No text content detected]' else 'No text'}")
                print(f"  Tables: {len(page.tables)}")
                print(f"  Images: {len(page.images)}")
                print(f"  Confidence: {page.confidence_score}")
                
                if page.tables:
                    print(f"  Table content: {page.tables[0]['content'][:50]}...")
                    
            except Exception as e:
                print(f"✗ Error processing page {page_num}: {e}")
                
            print("-" * 80)

if __name__ == "__main__":
    test_doctags_processing()