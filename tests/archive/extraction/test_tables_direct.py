#!/usr/bin/env python3
"""
Direct table extraction test for pages 2 and 6
"""

import pdfplumber
from pathlib import Path

def extract_tables_directly():
    """Extract tables directly with pdfplumber"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    print("="*80)
    print("📊 DIREKTE TABELLEN-EXTRAKTION MIT PDFPLUMBER")
    print("="*80)
    
    with pdfplumber.open(pdf_path) as pdf:
        # Page 2
        print("\n" + "="*80)
        print("📄 SEITE 2 - MOTORISIERUNG")
        print("="*80)
        
        page2 = pdf.pages[1]  # 0-indexed
        tables2 = page2.extract_tables()
        
        print(f"\nAnzahl Tabellen gefunden: {len(tables2)}")
        
        if tables2:
            for i, table in enumerate(tables2):
                print(f"\n--- TABELLE {i+1} ---")
                print(f"Größe: {len(table)} Zeilen x {len(table[0]) if table else 0} Spalten")
                
                print("\n📋 VOLLSTÄNDIGE TABELLE:")
                for row_idx, row in enumerate(table):
                    print(f"Zeile {row_idx}: {row}")
        
        # Also show raw text
        print("\n📝 RAW TEXT (erste 600 Zeichen):")
        print("-"*40)
        text2 = page2.extract_text()
        print(text2[:600] + "..." if len(text2) > 600 else text2)
        
        # Page 6
        print("\n" + "="*80)
        print("📄 SEITE 6 - TECHNISCHE DATEN")
        print("="*80)
        
        page6 = pdf.pages[5]  # 0-indexed
        tables6 = page6.extract_tables()
        
        print(f"\nAnzahl Tabellen gefunden: {len(tables6)}")
        
        if tables6:
            for i, table in enumerate(tables6):
                print(f"\n--- TABELLE {i+1} ---")
                print(f"Größe: {len(table)} Zeilen x {len(table[0]) if table else 0} Spalten")
                
                print("\n📋 VOLLSTÄNDIGE TABELLE:")
                for row_idx, row in enumerate(table):
                    print(f"Zeile {row_idx}: {row}")
        
        # Also show raw text
        print("\n📝 RAW TEXT (erste 600 Zeichen):")
        print("-"*40)
        text6 = page6.extract_text()
        print(text6[:600] + "..." if len(text6) > 600 else text6)

if __name__ == "__main__":
    extract_tables_directly()