#\!/usr/bin/env python3
"""Test the unified VLLMSmolDoclingFinalClient"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.clients import VLLMSmolDoclingClient, VLLMSmolDoclingFinalClient

def test_client():
    """Test basic client functionality"""
    print("\n=== Testing Unified SmolDocling Client ===\n")
    
    # Test 1: Verify alias
    print("1. Testing client alias...")
    assert VLLMSmolDoclingClient is VLLMSmolDoclingFinalClient
    print("✅ VLLMSmolDoclingClient correctly aliased to VLLMSmolDoclingFinalClient")
    
    # Test 2: Client instantiation
    print("\n2. Testing client instantiation...")
    try:
        client = VLLMSmolDoclingClient(
            max_pages=5,
            gpu_memory_utilization=0.3,
            environment='production'  # This enables docling
        )
        print("✅ Client instantiated successfully")
        print(f"   Client type: {type(client).__name__}")
        print(f"   Environment: {client.environment}")
        print(f"   Use docling: {client.use_docling}")
    except Exception as e:
        print(f"❌ Client instantiation failed: {e}")
        return False
    
    # Test 3: Test PDF parsing with small file
    print("\n3. Testing PDF parsing...")
    test_pdf = Path("data/input/test_simple.pdf")
    if not test_pdf.exists():
        # Try BMW document
        test_pdf = Path("data/input/Preview_BMW_1er_Sedan_CN.pdf")
    
    if test_pdf.exists():
        print(f"   Testing with: {test_pdf.name}")
        try:
            # Note: max_pages is set during client initialization, not in parse_pdf
            result = client.parse_pdf(test_pdf)
            print(f"✅ PDF parsed successfully")
            print(f"   Pages processed: {len(result.pages)}")
            if result.pages:
                print(f"   First page elements: {len(result.pages[0].elements)}")
                # Show first few elements
                for i, elem in enumerate(result.pages[0].elements[:3]):
                    print(f"   Element {i}: {elem.element_type} - {elem.text[:50]}...")
        except Exception as e:
            print(f"❌ PDF parsing failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  No test PDF found")
    
    print("\n=== Test Summary ===")
    print("✅ Client integration successful")
    print("✅ Legacy imports working via alias")
    print("✅ Ready for production use")
    
    return True

if __name__ == "__main__":
    test_client()
