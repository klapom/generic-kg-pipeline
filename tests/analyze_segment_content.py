#!/usr/bin/env python3
"""
Analyze segment content for image tags
"""

import re
from pathlib import Path

# Read the HTML report to check segment content
html_file = Path("bmw_vlm_report_Preview_BMW_1er_Sedan_CN_20250718_093029.html")

if html_file.exists():
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for image tags in the HTML
    image_pattern = r'&lt;image&gt;.*?&lt;/image&gt;'
    figure_pattern = r'&lt;figure&gt;.*?&lt;/figure&gt;'
    
    images = re.findall(image_pattern, content)
    figures = re.findall(figure_pattern, content)
    
    print(f"Found {len(images)} image tags in HTML")
    print(f"Found {len(figures)} figure tags in HTML")
    
    if images:
        print("\nFirst few image tags:")
        for img in images[:5]:
            # Extract coordinates
            loc_pattern = r'&lt;loc_(\d+)&gt;'
            coords = re.findall(loc_pattern, img)
            if len(coords) >= 4:
                print(f"  Image at coordinates: {coords[:4]}")
    
    # Look for segments with many loc tags
    segment_pattern = r'<div class="segment-content">(.*?)</div>'
    segments = re.findall(segment_pattern, content, re.DOTALL)
    
    print(f"\nFound {len(segments)} segments")
    
    for i, seg in enumerate(segments):
        loc_count = len(re.findall(r'&lt;loc_\d+&gt;', seg))
        if loc_count > 20:
            print(f"\nSegment {i} has {loc_count} loc tags")
            # Check if it's a table or image
            if '&lt;table&gt;' in seg:
                print("  -> Contains table tag")
            if '&lt;image&gt;' in seg:
                print("  -> Contains image tag")
            if '&lt;figure&gt;' in seg:
                print("  -> Contains figure tag")