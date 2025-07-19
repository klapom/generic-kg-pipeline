#!/usr/bin/env python3
"""
Analyze docling output in HTML report
"""

import re
from pathlib import Path

# Find latest HTML report
html_files = list(Path(".").glob("bmw_vlm_report_*_094424.html"))
if not html_files:
    print("No recent HTML report found")
    exit(1)

html_file = sorted(html_files)[-1]
print(f"Analyzing: {html_file}")

with open(html_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract segment content to check what SmolDocling produced
segment_pattern = r'<div class="segment-content">(.*?)</div>'
segments = re.findall(segment_pattern, content, re.DOTALL)

print(f"\nFound {len(segments)} segments")

# Analyze segments for visual indicators
for i, seg in enumerate(segments):
    # Count various tags
    tag_pattern = r'&lt;([^&\s/]+)&gt;'
    tags = re.findall(tag_pattern, seg)
    unique_tags = set(tags)
    
    # Skip if only basic tags
    if unique_tags.issubset({'paragraph', 'text', 'section_header', 'title', 'loc_0', 'loc_1', 'loc_2', 'loc_3', 'loc_4', 'loc_5', 'loc_6', 'loc_7', 'loc_8', 'loc_9'}):
        continue
    
    print(f"\nSegment {i+1}:")
    print(f"  Unique tags: {unique_tags}")
    
    # Look for picture tags
    if 'picture' in unique_tags:
        print("  ✅ Contains picture tag!")
        picture_pattern = r'&lt;picture&gt;(.*?)&lt;/picture&gt;'
        pictures = re.findall(picture_pattern, seg, re.DOTALL)
        print(f"  Found {len(pictures)} pictures")
    
    # Look for table tags  
    if 'table' in unique_tags:
        print("  📊 Contains table tag")
    
    # Look for figure tags
    if 'figure' in unique_tags:
        print("  🖼️ Contains figure tag")
    
    # Show preview if interesting
    if not unique_tags.issubset({'paragraph', 'text', 'loc_0', 'loc_1', 'loc_2', 'loc_3', 'loc_4', 'loc_5', 'loc_6', 'loc_7', 'loc_8', 'loc_9'}):
        preview = seg[:300].replace('\n', ' ')
        print(f"  Preview: {preview}...")