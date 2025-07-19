#!/usr/bin/env python3
"""
Check for all possible visual element tags in SmolDocling output
"""

import re
from pathlib import Path

# Check the HTML report for all possible visual tags
html_file = Path("bmw_vlm_report_Preview_BMW_1er_Sedan_CN_20250718_093029.html")

if html_file.exists():
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for various possible tag patterns (escaped in HTML)
    patterns = [
        (r'&lt;image&gt;.*?&lt;/image&gt;', 'image'),
        (r'&lt;picture&gt;.*?&lt;/picture&gt;', 'picture'),
        (r'&lt;figure&gt;.*?&lt;/figure&gt;', 'figure'),
        (r'&lt;img&gt;.*?&lt;/img&gt;', 'img'),
        (r'&lt;graphic&gt;.*?&lt;/graphic&gt;', 'graphic'),
        (r'&lt;visual&gt;.*?&lt;/visual&gt;', 'visual'),
        (r'&lt;diagram&gt;.*?&lt;/diagram&gt;', 'diagram'),
        (r'&lt;chart&gt;.*?&lt;/chart&gt;', 'chart'),
        (r'&lt;illustration&gt;.*?&lt;/illustration&gt;', 'illustration'),
    ]
    
    print("Checking for visual element tags in segments:")
    print("=" * 50)
    
    for pattern, tag_name in patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"\nFound {len(matches)} <{tag_name}> tags!")
            # Show first match
            print(f"Example: {matches[0][:100]}...")
    
    # Also check for any tags with 'pic' or 'img' in them
    any_pic_pattern = r'&lt;[^&]*pic[^&]*&gt;'
    any_img_pattern = r'&lt;[^&]*img[^&]*&gt;'
    
    pic_tags = re.findall(any_pic_pattern, content)
    img_tags = re.findall(any_img_pattern, content)
    
    if pic_tags:
        print(f"\nFound tags containing 'pic': {set(pic_tags[:5])}")
    if img_tags:
        print(f"\nFound tags containing 'img': {set(img_tags[:5])}")
    
    # Check for segments with many loc tags that might be visual elements
    segment_pattern = r'<div class="segment-content">(.*?)</div>'
    segments = re.findall(segment_pattern, content, re.DOTALL)
    
    print(f"\n\nAnalyzing {len(segments)} segments for potential visual elements:")
    print("=" * 50)
    
    for i, seg in enumerate(segments):
        loc_count = len(re.findall(r'&lt;loc_\d+&gt;', seg))
        if loc_count > 10:  # Lower threshold
            print(f"\nSegment {i} has {loc_count} loc tags")
            
            # Check what other tags are present
            all_tags = re.findall(r'&lt;([^&\s]+?)&gt;', seg)
            unique_tags = set(all_tags)
            print(f"Unique tags in segment: {unique_tags}")
            
            # Show a snippet
            snippet = seg[:200].replace('\n', ' ')
            print(f"Content preview: {snippet}...")