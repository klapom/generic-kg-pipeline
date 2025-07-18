#!/usr/bin/env python3
"""
Quick test to verify all fixes are working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("🔧 Verifying Fixes")
print("="*60)

# Test 1: Import checks
print("\n1. Testing imports...")
try:
    from core.clients.vllm_smoldocling_final import VLLMSmolDoclingFinalClient
    print("   ✅ VLLMSmolDoclingFinalClient imports successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")

# Test 2: Check method signatures
print("\n2. Testing method signatures...")
try:
    client = VLLMSmolDoclingFinalClient(environment="production")
    
    # Check if _format_table exists and has correct signature
    import inspect
    sig = inspect.signature(client._format_table)
    params = list(sig.parameters.keys())
    if params == ['self', 'data']:
        print("   ✅ _format_table has correct parameter 'data'")
    else:
        print(f"   ❌ _format_table has wrong parameters: {params}")
        
    # Check if _check_and_fix_repetitions exists
    if hasattr(client, '_check_and_fix_repetitions'):
        print("   ✅ _check_and_fix_repetitions method exists")
    else:
        print("   ❌ _check_and_fix_repetitions method missing")
        
except Exception as e:
    print(f"   ❌ Method check error: {e}")

# Test 3: Test repetition detection
print("\n3. Testing repetition detection...")
try:
    client = VLLMSmolDoclingFinalClient(environment="production")
    
    # Test with repetitive content (but with enough content before repetition)
    test_doctags = """<doctag>
<section_header><loc_50><loc_50><loc_450><loc_80>Test Header</section_header>
<text><loc_50><loc_90><loc_450><loc_120>Normal content line 1</text>
<text><loc_50><loc_130><loc_450><loc_160>Normal content line 2</text>
<text><loc_50><loc_170><loc_450><loc_200>Normal content line 3</text>
<text><loc_50><loc_210><loc_450><loc_240>Repeated line</text>
<text><loc_50><loc_250><loc_450><loc_280>Repeated line</text>
<text><loc_50><loc_290><loc_450><loc_320>Repeated line</text>
<text><loc_50><loc_330><loc_450><loc_360>Repeated line</text>
<text><loc_50><loc_370><loc_450><loc_400>Repeated line</text>
<text><loc_50><loc_410><loc_450><loc_440>Repeated line</text>
<text><loc_50><loc_450><loc_450><loc_480>Repeated line</text>
<text><loc_50><loc_490><loc_450><loc_520>Repeated line</text>
<text><loc_50><loc_530><loc_450><loc_560>Repeated line</text>
<text><loc_50><loc_570><loc_450><loc_600>Repeated line</text>
<text><loc_50><loc_610><loc_450><loc_640>Repeated line</text>
<text><loc_50><loc_650><loc_450><loc_680>Repeated line</text>
</doctag>"""
    
    result = client._check_and_fix_repetitions(test_doctags)
    if "WARNING: Output truncated" in result:
        print("   ✅ Repetition detection adds warning comment")
    else:
        print("   ❌ Warning comment not added")
        
    # Test with early repetition (should raise error)
    test_early_repetition = """<doctag>
<text>Only one line</text>
""" + "<text>Repeated</text>\n" * 20
    
    try:
        result = client._check_and_fix_repetitions(test_early_repetition)
        print("   ❌ Early repetition should raise ParseError")
    except Exception as e:
        if "repetition bug detected too early" in str(e):
            print("   ✅ Early repetition correctly raises ParseError")
        else:
            print(f"   ❌ Wrong error: {e}")
            
except Exception as e:
    print(f"   ❌ Repetition test error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import test for test script
print("\n4. Testing test script imports...")
try:
    # Just check if re is available in the test context
    import re
    print("   ✅ 're' module available")
except:
    print("   ❌ 're' module not available")

print("\n" + "="*60)
print("✨ Fix verification complete!")
print("="*60)