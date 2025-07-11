#!/usr/bin/env python3
"""
Debug table separation issue
"""

import pdfplumber
from pathlib import Path
from core.parsers.table_text_separator import TableTextSeparator

def debug_separation():
    """Debug why table separation isn't working"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    separator = TableTextSeparator()
    
    print("="*80)
    print("🔍 DEBUGGING TABLE SEPARATION")
    print("="*80)
    
    with pdfplumber.open(pdf_path) as pdf:
        # Test page 2
        page = pdf.pages[1]
        text = page.extract_text()
        tables = page.extract_tables()
        
        print(f"\n📄 PAGE 2 Analysis:")
        print(f"Text length: {len(text)}")
        print(f"Tables found: {len(tables)}")
        
        if tables:
            # Convert to expected format
            formatted_tables = []
            for i, table in enumerate(tables):
                formatted_table = {
                    'table_id': i + 1,
                    'headers': table[1] if len(table) > 1 else [],  # Second row has headers
                    'data': table[2:] if len(table) > 2 else [],    # Rest is data
                    'row_count': len(table) - 2 if len(table) > 2 else 0,
                    'col_count': len(table[0]) if table else 0
                }
                formatted_tables.append(formatted_table)
                
                print(f"\nTable {i+1} structure:")
                print(f"  Total rows: {len(table)}")
                print(f"  First row: {table[0]}")
                print(f"  Second row (headers): {table[1] if len(table) > 1 else 'N/A'}")
            
            # Test separation
            print("\n🔬 Testing separation...")
            result = separator.separate_content(text, formatted_tables)
            
            print(f"\nSeparation results:")
            print(f"  Original text length: {len(text)}")
            print(f"  Pure text length: {len(result['pure_text'])}")
            print(f"  Characters removed: {len(text) - len(result['pure_text'])}")
            print(f"  Table regions found: {len(result['table_regions'])}")
            
            if result['table_regions']:
                print(f"\nTable boundaries:")
                for i, (start, end) in enumerate(result['table_regions']):
                    print(f"  Region {i+1}: {start}-{end} ({end-start} chars)")
                    print(f"    Content: '{text[start:min(start+50, end)]}...'")
            
            print(f"\n📝 PURE TEXT (first 500 chars):")
            print("-"*40)
            print(result['pure_text'][:500] + "..." if len(result['pure_text']) > 500 else result['pure_text'])

if __name__ == "__main__":
    debug_separation()