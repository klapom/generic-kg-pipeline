#!/usr/bin/env python3
"""
Check for picture tags in segments - based on SmolDocling documentation
"""

import re
from pathlib import Path

# Check the HTML report for picture tags
html_file = Path("bmw_vlm_report_Preview_BMW_1er_Sedan_CN_20250718_093029.html")

if html_file.exists():
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # SmolDocling uses <picture> tags according to documentation
    picture_pattern = r'&lt;picture&gt;.*?&lt;/picture&gt;'
    
    pictures = re.findall(picture_pattern, content, re.DOTALL)
    
    print(f"Found {len(pictures)} <picture> tags in HTML")
    
    if pictures:
        print("\nFirst few picture tags:")
        for i, pic in enumerate(pictures[:3]):
            print(f"\nPicture {i+1}:")
            print(pic[:200] + "...")
            
            # Extract coordinates
            loc_pattern = r'&lt;loc_(\d+)&gt;'
            coords = re.findall(loc_pattern, pic)
            if len(coords) >= 4:
                print(f"Coordinates: x1={coords[0]}, y1={coords[1]}, x2={coords[2]}, y2={coords[3]}")
    
    # Also check for any picture-related patterns
    print("\n" + "=" * 50)
    print("Checking for picture-related patterns:")
    
    # Check if 'picture' appears anywhere in escaped form
    picture_count = content.count('&lt;picture')
    print(f"\nOccurrences of '&lt;picture': {picture_count}")
    
    # Check raw segment content
    segment_pattern = r'<div class="segment-content">(.*?)</div>'
    segments = re.findall(segment_pattern, content, re.DOTALL)
    
    for i, seg in enumerate(segments):
        if 'picture' in seg.lower():
            print(f"\nSegment {i} contains 'picture':")
            # Find context around 'picture'
            idx = seg.lower().find('picture')
            start = max(0, idx - 50)
            end = min(len(seg), idx + 50)
            print(f"Context: ...{seg[start:end]}...")
else:
    print(f"HTML file not found: {html_file}")