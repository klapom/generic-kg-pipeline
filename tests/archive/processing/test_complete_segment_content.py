#!/usr/bin/env python3
"""
Show COMPLETE content of each segment for 1:1 verification
"""

import logging
from pathlib import Path
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor
from plugins.parsers.base_parser import Segment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for better readability
)
logger = logging.getLogger(__name__)

def show_complete_segments():
    """Show complete segment content without truncation"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    print("="*100)
    print("🔍 VOLLSTÄNDIGER SEGMENT-INHALT FÜR 1:1 PRÜFUNG")
    print("="*100)
    
    # Initialize advanced extractor with layout
    layout_settings = {
        'use_layout': True,
        'table_x_tolerance': 3,
        'table_y_tolerance': 3,
        'text_x_tolerance': 5,
        'text_y_tolerance': 5
    }
    extractor = AdvancedPDFExtractor(layout_settings)
    
    # Process pages 2 and 6
    for page_num in [2, 6]:
        print(f"\n{'='*100}")
        print(f"📄 SEITE {page_num} - VOLLSTÄNDIGE SEGMENTE")
        print(f"{'='*100}")
        
        # Extract with bbox filtering
        result = extractor.extract_page_with_bbox(pdf_path, page_num)
        
        # Print extraction summary
        print(f"\n📊 EXTRAKTIONS-ZUSAMMENFASSUNG:")
        print(f"  - Tabellen gefunden: {result['metadata']['tables_found']}")
        print(f"  - Text außerhalb Tabellen: {result['metadata']['text_length']} Zeichen")
        print(f"  - Volltext der Seite: {result['metadata']['full_text_length']} Zeichen")
        
        # Create and show segments
        segments = []
        
        # TEXT SEGMENT (filtered content)
        if result['text'].strip():
            text_segment = Segment(
                content=result['text'],
                segment_type='text',
                metadata={
                    'page_number': page_num,
                    'content_type': 'text_outside_tables',
                    'char_count': len(result['text'])
                }
            )
            segments.append(text_segment)
            
            print(f"\n{'='*80}")
            print(f"📝 SEGMENT 1: TEXT (außerhalb Tabellen)")
            print(f"{'='*80}")
            print(f"Länge: {len(text_segment.content)} Zeichen")
            print(f"{'─'*80}")
            print("VOLLSTÄNDIGER INHALT:")
            print("─"*80)
            print(text_segment.content)
            print("─"*80)
            print("[ENDE TEXT-SEGMENT]")
        
        # TABLE SEGMENTS
        for i, table in enumerate(result['tables']):
            # Use layout-preserved text if available
            if table.get('layout_text'):
                table_content = table['layout_text']
            else:
                # Create simple representation
                lines = []
                # Add headers
                if table['headers']:
                    lines.append(" | ".join(str(h) for h in table['headers']))
                    lines.append("-" * 100)
                # Add data rows
                for row in table['data']:
                    lines.append(" | ".join(str(cell) if cell else "" for cell in row))
                table_content = "\n".join(lines)
            
            table_segment = Segment(
                content=table_content,
                segment_type='table',
                metadata={
                    'page_number': page_num,
                    'table_id': table['table_id'],
                    'table_type': table['table_type'],
                    'row_count': table['row_count'],
                    'col_count': table['col_count'],
                    'bbox': table['bbox']
                }
            )
            segments.append(table_segment)
            
            print(f"\n{'='*80}")
            print(f"📊 SEGMENT {i+2}: TABELLE {table['table_id']}")
            print(f"{'='*80}")
            print(f"Typ: {table['table_type']}")
            print(f"Größe: {table['row_count']} Zeilen x {table['col_count']} Spalten")
            print(f"Länge: {len(table_segment.content)} Zeichen")
            print(f"BBox: {table['bbox']}")
            print(f"{'─'*80}")
            print("VOLLSTÄNDIGER INHALT:")
            print("─"*80)
            print(table_segment.content)
            print("─"*80)
            print("[ENDE TABELLEN-SEGMENT]")
        
        # Show structured data for tables
        print(f"\n{'='*80}")
        print("📋 STRUKTURIERTE TABELLENDATEN (für Triple-Generierung):")
        print(f"{'='*80}")
        
        for i, table in enumerate(result['tables']):
            print(f"\nTabelle {table['table_id']} - Strukturierte Daten:")
            print(f"Headers: {table['headers']}")
            print(f"\nDatenzeilen:")
            for j, row in enumerate(table['data']):
                print(f"  Zeile {j+1}: {row}")
        
        print(f"\n{'='*100}")
        print(f"ENDE SEITE {page_num}")
        print(f"{'='*100}")

if __name__ == "__main__":
    show_complete_segments()