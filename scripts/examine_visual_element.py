#!/usr/bin/env python3
"""
Simple script to examine visual element structure from JSON report
"""
import json

# Load the analysis report
with open('visual_elements_analysis_20250718_081840.json', 'r') as f:
    data = json.load(f)

print("Visual Elements Analysis Summary")
print("=" * 60)

# Without VLM
print("\n1. WITHOUT VLM:")
print(f"   Total segments: {data['without_vlm']['total_segments']}")
print(f"   Total visual elements: {data['without_vlm']['total_visual_elements']}")
print(f"   Visual element types: {data['without_vlm']['visual_element_types']}")

# With VLM
print("\n2. WITH VLM:")
print(f"   Total segments: {data['with_vlm']['total_segments']}")
print(f"   Total visual elements: {data['with_vlm']['total_visual_elements']}")
print(f"   VLM analyzed: {data['with_vlm']['vlm_analyzed']}")

# Examine first visual element
if data['with_vlm']['visual_elements']:
    print("\n3. FIRST VISUAL ELEMENT STRUCTURE:")
    ve = data['with_vlm']['visual_elements'][0]
    
    print(f"   - element_type: {ve.get('element_type')}")
    print(f"   - source_format: {ve.get('source_format')}")
    print(f"   - content_hash: {ve.get('content_hash')}")
    print(f"   - confidence: {ve.get('confidence')}")
    print(f"   - page_or_slide: {ve.get('page_or_slide')}")
    print(f"   - segment_reference: {ve.get('segment_reference')}")
    print(f"   - bounding_box: {ve.get('bounding_box')}")
    print(f"   - extracted_data: {ve.get('extracted_data')}")
    print(f"   - analysis_metadata: {ve.get('analysis_metadata')}")
    
    # Check VLM description
    vlm_desc = ve.get('vlm_description')
    if vlm_desc:
        if vlm_desc.startswith("Analysis failed"):
            print(f"   - vlm_description: FAILED - {vlm_desc[:100]}...")
        else:
            print(f"   - vlm_description: FILLED ({len(vlm_desc)} chars)")
    else:
        print(f"   - vlm_description: EMPTY")

# Count successful VLM analyses
print("\n4. VLM ANALYSIS RESULTS:")
successful = 0
failed = 0
empty = 0

for ve in data['with_vlm']['visual_elements']:
    vlm_desc = ve.get('vlm_description')
    if not vlm_desc:
        empty += 1
    elif vlm_desc.startswith("Analysis failed"):
        failed += 1
    else:
        successful += 1

print(f"   - Successful: {successful}")
print(f"   - Failed: {failed}")  
print(f"   - Empty: {empty}")
print(f"   - Total: {len(data['with_vlm']['visual_elements'])}")