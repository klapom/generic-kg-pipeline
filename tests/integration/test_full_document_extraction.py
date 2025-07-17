#!/usr/bin/env python3
"""
Extract and show COMPLETE content for entire BMW document
"""

import logging
from pathlib import Path
from core.parsers.advanced_pdf_extractor import AdvancedPDFExtractor
from core.parsers import Segment
import pdfplumber

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for better readability
)
logger = logging.getLogger(__name__)

def extract_full_document():
    """Extract complete content from entire BMW document"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    # Get total page count
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    print("="*100)
    print(f"🔍 VOLLSTÄNDIGE DOKUMENT-EXTRAKTION: {pdf_path.name}")
    print(f"📄 Gesamt-Seitenanzahl: {total_pages}")
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
    
    # Process all pages
    all_segments = []
    
    for page_num in range(1, total_pages + 1):
        print(f"\n{'='*100}")
        print(f"📄 SEITE {page_num} von {total_pages}")
        print(f"{'='*100}")
        
        try:
            # Extract with bbox filtering
            result = extractor.extract_page_with_bbox(pdf_path, page_num)
            
            # Print extraction summary
            print(f"\n📊 EXTRAKTIONS-ZUSAMMENFASSUNG:")
            print(f"  - Tabellen gefunden: {result['metadata']['tables_found']}")
            print(f"  - Text außerhalb Tabellen: {result['metadata']['text_length']} Zeichen")
            print(f"  - Volltext der Seite: {result['metadata']['full_text_length']} Zeichen")
            print(f"  - Tabellen-Abdeckung: {result['metadata']['table_coverage']}%")
            
            # Create segments for this page
            page_segments = []
            
            # TEXT SEGMENT (filtered content)
            if result['text'].strip() and len(result['text'].strip()) > 20:
                text_segment = Segment(
                    content=result['text'],
                    segment_type='text',
                    metadata={
                        'page_number': page_num,
                        'content_type': 'text_outside_tables',
                        'char_count': len(result['text'])
                    }
                )
                page_segments.append(text_segment)
                all_segments.append(text_segment)
                
                print(f"\n{'='*80}")
                print(f"📝 TEXT-SEGMENT (außerhalb Tabellen)")
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
                page_segments.append(table_segment)
                all_segments.append(table_segment)
                
                print(f"\n{'='*80}")
                print(f"📊 TABELLE {i+1} (ID: {table['table_id']})")
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
                print(f"\n📋 STRUKTURIERTE DATEN (für Triple-Generierung):")
                print(f"Headers: {table['headers']}")
                if len(table['data']) > 0:
                    print(f"Erste Datenzeile: {table['data'][0]}")
                    print(f"Letzte Datenzeile: {table['data'][-1]}")
                    print(f"Gesamt Zeilen: {len(table['data'])}")
            
            # Page summary
            print(f"\n{'─'*80}")
            print(f"📋 SEITE {page_num} ZUSAMMENFASSUNG:")
            print(f"  - Segmente erstellt: {len(page_segments)}")
            print(f"  - Text-Segmente: {sum(1 for s in page_segments if s.segment_type == 'text')}")
            print(f"  - Tabellen-Segmente: {sum(1 for s in page_segments if s.segment_type == 'table')}")
            
            # If no content found
            if not page_segments:
                print(f"  - ⚠️  Keine relevanten Inhalte auf dieser Seite gefunden")
        
        except Exception as e:
            print(f"\n❌ FEHLER bei Seite {page_num}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*100}")
        print(f"ENDE SEITE {page_num}")
        print(f"{'='*100}")
    
    # Final summary
    print(f"\n{'='*100}")
    print("📊 GESAMT-ZUSAMMENFASSUNG DES DOKUMENTS")
    print(f"{'='*100}")
    print(f"  - Seiten verarbeitet: {total_pages}")
    print(f"  - Segmente gesamt: {len(all_segments)}")
    print(f"  - Text-Segmente: {sum(1 for s in all_segments if s.segment_type == 'text')}")
    print(f"  - Tabellen-Segmente: {sum(1 for s in all_segments if s.segment_type == 'table')}")
    print(f"  - Durchschnittliche Segment-Länge: {sum(len(s.content) for s in all_segments) / len(all_segments):.0f} Zeichen" if all_segments else "N/A")
    
    # Show segment distribution by page
    print(f"\n📄 SEGMENT-VERTEILUNG PRO SEITE:")
    for page in range(1, total_pages + 1):
        page_segs = [s for s in all_segments if s.metadata.get('page_number') == page]
        if page_segs:
            text_count = sum(1 for s in page_segs if s.segment_type == 'text')
            table_count = sum(1 for s in page_segs if s.segment_type == 'table')
            total_chars = sum(len(s.content) for s in page_segs)
            print(f"  - Seite {page}: {len(page_segs)} Segmente ({text_count} Text, {table_count} Tabellen) = {total_chars} Zeichen")

if __name__ == "__main__":
    extract_full_document()