#!/usr/bin/env python3
"""
Test structured table extraction for BMW page 2
Goal: Extract tables with structure for triple generation
"""

import logging
from pathlib import Path
import json
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_structured_extraction():
    """Test different approaches for structured table extraction"""
    
    pdf_path = Path("data/input/Preview_BMW_3er_G20.pdf")
    
    logger.info("="*80)
    logger.info("🎯 TESTING STRUCTURED TABLE EXTRACTION FOR TRIPLES")
    logger.info("="*80)
    
    # 1. Test with pdfplumber (better for tables)
    try:
        import pdfplumber
        logger.info("\n1️⃣ PDFPLUMBER STRUCTURED EXTRACTION:")
        logger.info("-"*40)
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[1]  # Page 2 (0-indexed)
            
            # Extract tables
            tables = page.extract_tables()
            
            if tables:
                logger.info(f"✅ Found {len(tables)} tables!")
                
                for i, table in enumerate(tables):
                    logger.info(f"\n📊 Table {i+1} ({len(table)} rows x {len(table[0]) if table else 0} columns):")
                    
                    # Assume first row is header
                    if len(table) > 1:
                        headers = table[0]
                        logger.info(f"Headers: {headers}")
                        
                        # Generate triples from table
                        triples = []
                        for row_idx, row in enumerate(table[1:], 1):
                            entity = row[0] if row else "Unknown"  # First column as entity
                            
                            for col_idx, value in enumerate(row[1:], 1):
                                if col_idx < len(headers) and value:
                                    predicate = headers[col_idx]
                                    triple = (entity, predicate, value)
                                    triples.append(triple)
                                    
                        logger.info(f"\n🔗 Generated {len(triples)} triples:")
                        for triple in triples[:5]:  # Show first 5
                            logger.info(f"   ({triple[0]}, {triple[1]}, {triple[2]})")
            else:
                logger.info("❌ No tables found by pdfplumber")
                
    except ImportError:
        logger.info("pdfplumber not available, installing...")
        import subprocess
        subprocess.run(["pip", "install", "pdfplumber"], check=True)
        logger.info("Please run the script again")
        return
    
    # 2. Alternative: Parse text with regex patterns
    logger.info("\n2️⃣ REGEX-BASED STRUCTURE EXTRACTION:")
    logger.info("-"*40)
    
    # Get text from PyPDF2
    import PyPDF2
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        page = reader.pages[1]
        text = page.extract_text()
    
    # Extract Motorisierungen table with regex
    motorisierungen_pattern = r'Motorisierungen\s*(.*?)(?=›|$)'
    match = re.search(motorisierungen_pattern, text, re.DOTALL)
    
    if match:
        table_text = match.group(1)
        
        # Parse rows - look for patterns like "320d R4 2,0l Turbo 8G-AT 140 / 400 115 –122"
        row_pattern = r'(\w+)\s+(R\d+)\s+([\d,]+l)\s+(\w+)\s+([\w-]+)\s+([\d\s/]+)\s+([\d\s–-]+)'
        rows = re.findall(row_pattern, table_text)
        
        if rows:
            logger.info(f"✅ Extracted {len(rows)} rows from Motorisierungen table")
            
            # Define column names
            columns = ["Modell", "Motor", "Hubraum", "Aufladung", "Getriebe", "Leistung/Drehmoment", "CO2-Emission"]
            
            # Generate structured data
            structured_data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                structured_data.append(row_dict)
                
            logger.info("\n📋 Structured data:")
            for data in structured_data[:3]:  # Show first 3
                logger.info(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Generate triples
            logger.info("\n🔗 Generated triples (Subject, Predicate, Object):")
            for data in structured_data:
                model = data["Modell"]
                for key, value in data.items():
                    if key != "Modell" and value:
                        logger.info(f"   (BMW {model}, hat_{key}, {value})")
    
    # 3. Alternative: Camelot (specialized for tables)
    logger.info("\n3️⃣ CAMELOT TABLE EXTRACTION (if available):")
    logger.info("-"*40)
    
    try:
        import camelot
        
        # Extract tables from page 2
        tables = camelot.read_pdf(str(pdf_path), pages='2', flavor='stream')
        
        if len(tables) > 0:
            logger.info(f"✅ Found {len(tables)} tables with Camelot")
            
            for i, table in enumerate(tables):
                logger.info(f"\n📊 Table {i+1}:")
                df = table.df
                logger.info(f"Shape: {df.shape}")
                logger.info(f"First few rows:\n{df.head()}")
                
                # Generate triples from DataFrame
                if len(df) > 1:
                    headers = df.iloc[0].tolist()
                    for idx, row in df.iloc[1:].iterrows():
                        entity = row[0]
                        for col_idx, value in enumerate(row[1:], 1):
                            if col_idx < len(headers) and value:
                                predicate = headers[col_idx]
                                logger.info(f"   ({entity}, {predicate}, {value})")
        else:
            logger.info("❌ No tables found by Camelot")
            
    except ImportError:
        logger.info("Camelot not available (requires system dependencies)")
    except Exception as e:
        logger.info(f"Camelot error: {e}")
    
    # 4. Custom parser for specific format
    logger.info("\n4️⃣ CUSTOM PARSER FOR BMW TABLE FORMAT:")
    logger.info("-"*40)
    
    # Parse the specific Motorisierungen table format
    lines = text.split('\n')
    
    in_motor_section = False
    motor_data = []
    
    for line in lines:
        if 'Motorisierungen' in line:
            in_motor_section = True
            continue
        elif in_motor_section and ('›' in line or 'Verfügbar' in line):
            break
        elif in_motor_section and line.strip():
            # Try to parse motor specification lines
            # Format: Model Motor Type Power/Torque CO2 Price
            parts = line.split()
            if len(parts) >= 6 and parts[0] in ['320i', '330i', 'M340i', '318d', '320d', '330d']:
                motor_entry = {
                    'model': parts[0],
                    'motor_type': ' '.join(parts[1:3]),
                    'details': ' '.join(parts[3:])
                }
                motor_data.append(motor_entry)
    
    if motor_data:
        logger.info(f"✅ Parsed {len(motor_data)} motor specifications")
        
        # Generate semantic triples
        logger.info("\n🔗 Semantic triples for Knowledge Graph:")
        for entry in motor_data:
            model_uri = f"bmw:3er_{entry['model']}"
            logger.info(f"\n{model_uri}:")
            logger.info(f"   - rdf:type → bmw:VehicleModel")
            logger.info(f"   - bmw:modelDesignation → '{entry['model']}'")
            logger.info(f"   - bmw:hasMotor → '{entry['motor_type']}'")
            logger.info(f"   - bmw:technicalDetails → '{entry['details']}'")
    
    # Save structured results
    output = {
        'extraction_methods': {
            'pdfplumber': 'Best for table structure',
            'regex': 'Good for known patterns',
            'camelot': 'Excellent for complex tables',
            'custom': 'Best for specific formats'
        },
        'motor_data': motor_data,
        'sample_triples': [
            ('bmw:3er_320d', 'rdf:type', 'bmw:VehicleModel'),
            ('bmw:3er_320d', 'bmw:hasMotor', 'R4 2,0l Turbo'),
            ('bmw:3er_320d', 'bmw:hasPower', '140 kW'),
            ('bmw:3er_320d', 'bmw:hasCO2Emission', '115-122 g/km')
        ]
    }
    
    with open('data/output/structured_table_extraction.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📝 Results saved to: data/output/structured_table_extraction.json")
    
    logger.info("\n" + "="*80)
    logger.info("📊 RECOMMENDATION FOR TRIPLE GENERATION:")
    logger.info("="*80)
    logger.info("1. Use pdfplumber for general table extraction")
    logger.info("2. Apply custom parsers for known table formats")
    logger.info("3. Generate RDF triples with proper ontology")
    logger.info("4. Store in Fuseki triple store for SPARQL queries")

if __name__ == "__main__":
    test_structured_extraction()