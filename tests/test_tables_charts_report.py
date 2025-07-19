#!/usr/bin/env python3
"""
Test HTML report generation with tables and charts
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_qwen25_html_report import test_bmw_with_html_report


async def test_tables_charts():
    """Test with our generated PDF containing tables and charts"""
    
    # Temporarily rename the test PDF to match BMW pattern
    test_pdf = Path("data/input/test_tables_charts.pdf")
    bmw_test_pdf = Path("data/input/BMW_Tables_Charts_Test.pdf")
    
    if test_pdf.exists():
        # Copy file
        import shutil
        shutil.copy(test_pdf, bmw_test_pdf)
        
        print(f"Testing with: {bmw_test_pdf}")
        
        # Run the test
        report_path = await test_bmw_with_html_report()
        
        print(f"\nReport generated: {report_path}")
        print("\nThis report should contain:")
        print("- Tables rendered as HTML tables")
        print("- Charts analyzed by VLM")
        print("- Structured data extraction from tables")
        
        # Clean up
        bmw_test_pdf.unlink()
    else:
        print(f"Test PDF not found: {test_pdf}")


if __name__ == "__main__":
    asyncio.run(test_tables_charts())