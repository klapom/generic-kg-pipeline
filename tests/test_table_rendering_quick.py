#!/usr/bin/env python3
"""
Quick test to verify table rendering in HTML reports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_qwen25_html_report import render_table_content


def test_table_rendering():
    """Test various table formats"""
    
    # Test 1: Markdown table format (what we expect from fixed SmolDocling)
    markdown_table = """| Model | Engine | Power (HP) | Torque (Nm) | Price (EUR) |
| --- | --- | --- | --- | --- |
| BMW 320i | 2.0L Turbo | 184 | 300 | 45,900 |
| BMW 330i | 2.0L Turbo | 258 | 400 | 52,300 |
| BMW M340i | 3.0L Turbo | 387 | 500 | 68,500 |"""
    
    print("Test 1: Markdown table")
    result = render_table_content(markdown_table)
    if result:
        print("✅ Successfully rendered as HTML table:")
        print(result[:200] + "...")
    else:
        print("❌ Failed to render")
    
    # Test 2: Tab-separated table
    tab_table = """Model\tEngine\tPower (HP)\tTorque (Nm)\tPrice (EUR)
BMW 320i\t2.0L Turbo\t184\t300\t45,900
BMW 330i\t2.0L Turbo\t258\t400\t52,300"""
    
    print("\nTest 2: Tab-separated table")
    result = render_table_content(tab_table)
    if result:
        print("✅ Successfully rendered as HTML table")
    else:
        print("❌ Failed to render")
    
    # Test 3: Old TableCell format (should fail)
    tablecell_format = """TableCell(bbox=None, row_span=1, col_span=1, start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, text='Model')
TableCell(bbox=None, row_span=1, col_span=1, start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, text='Engine')"""
    
    print("\nTest 3: Old TableCell format")
    result = render_table_content(tablecell_format)
    if result:
        print("❌ Unexpectedly rendered (should fail)")
    else:
        print("✅ Correctly failed to render TableCell format")


if __name__ == "__main__":
    test_table_rendering()