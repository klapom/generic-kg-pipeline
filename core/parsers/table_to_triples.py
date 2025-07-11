"""
Table to Triple converter for Knowledge Graph generation
"""

import logging
import re
from typing import List, Dict, Tuple, Any
import pdfplumber
from pathlib import Path

logger = logging.getLogger(__name__)


class TableToTripleConverter:
    """
    Converts tables extracted from PDFs into semantic triples
    for knowledge graph generation
    """
    
    def __init__(self):
        self.namespace = "bmw:"  # Can be configured
        logger.info("Initialized TableToTripleConverter")
    
    def extract_motorisierung_table(self, pdf_path: Path, page_num: int = 2) -> List[Dict[str, Any]]:
        """
        Extract the Motorisierungen table from BMW PDF with proper structure
        """
        structured_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num - 1]
                
                # Try to find the Motorisierungen table
                tables = page.extract_tables()
                
                # Also get raw text for fallback parsing
                text = page.extract_text()
                
                # Method 1: Parse from pdfplumber tables
                if tables:
                    for table in tables:
                        # Skip if too few columns
                        if not table or len(table[0]) < 5:
                            continue
                            
                        # Look for Motorisierungen table
                        if any('Motor' in str(cell) for row in table for cell in row):
                            structured_data = self._parse_motor_table(table)
                            break
                
                # Method 2: Fallback to text parsing if no structured table found
                if not structured_data and text:
                    structured_data = self._parse_motor_text(text)
                    
        except Exception as e:
            logger.error(f"Error extracting table: {e}")
        
        return structured_data
    
    def _parse_motor_table(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        """Parse motor table from pdfplumber output"""
        data = []
        
        # Find header row (contains "Modell" or "Motor")
        header_idx = -1
        headers = []
        
        for i, row in enumerate(table):
            if any('Modell' in str(cell) or 'Motor' in str(cell) for cell in row if cell):
                header_idx = i
                # Clean headers
                headers = [self._clean_header(cell) for cell in row]
                break
        
        if header_idx == -1 or not headers:
            return data
        
        # Parse data rows
        for row in table[header_idx + 1:]:
            if not row or not row[0]:  # Skip empty rows
                continue
                
            # Create structured entry
            entry = {}
            for i, value in enumerate(row):
                if i < len(headers) and headers[i] and value:
                    entry[headers[i]] = self._clean_value(value)
            
            if entry.get('Modell'):  # Only add if we have a model
                data.append(entry)
        
        return data
    
    def _parse_motor_text(self, text: str) -> List[Dict[str, Any]]:
        """Fallback parser for text-based extraction"""
        data = []
        
        # Extract section between "Motorisierungen" and next section marker
        motor_section = re.search(r'Motorisierungen(.*?)(?:Verfügbar|›|$)', text, re.DOTALL)
        if not motor_section:
            return data
        
        section_text = motor_section.group(1)
        
        # Known BMW models pattern
        models = ['320i', '330i', 'M340i', '330e', '318d', '320d', '330d']
        
        for model in models:
            # Pattern: Model MotorType Transmission Power/Torque CO2 Price
            pattern = rf'{model}\s+([R]\d+\s+[\d,]+l\s+\w+)\s+([\w-]+)\s+([\d/\s]+)\s+([\d\s–-]+)'
            match = re.search(pattern, section_text)
            
            if match:
                entry = {
                    'Modell': model,
                    'Motor': match.group(1).strip(),
                    'Getriebe': match.group(2).strip(),
                    'Leistung_Drehmoment': match.group(3).strip(),
                    'CO2_Emission': match.group(4).strip()
                }
                
                # Parse power and torque
                power_torque = entry['Leistung_Drehmoment']
                power_match = re.search(r'(\d+)\s*/', power_torque)
                torque_match = re.search(r'/\s*(\d+)', power_torque)
                
                if power_match:
                    entry['Leistung_kW'] = power_match.group(1)
                if torque_match:
                    entry['Drehmoment_Nm'] = torque_match.group(1)
                
                data.append(entry)
        
        return data
    
    def generate_triples(self, motor_data: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Generate RDF triples from structured motor data
        
        Returns list of (subject, predicate, object) tuples
        """
        triples = []
        
        for entry in motor_data:
            model = entry.get('Modell', '')
            if not model:
                continue
            
            # Create URI for this vehicle model
            subject = f"{self.namespace}3er_{model}"
            
            # Type triple
            triples.append((subject, "rdf:type", f"{self.namespace}VehicleModel"))
            
            # Model designation
            triples.append((subject, f"{self.namespace}modelDesignation", f'"{model}"'))
            
            # Series
            triples.append((subject, f"{self.namespace}belongsToSeries", f"{self.namespace}3er_G20"))
            
            # Motor details
            if 'Motor' in entry:
                motor_uri = f"{self.namespace}motor_{model}_{entry['Motor'].replace(' ', '_')}"
                triples.append((subject, f"{self.namespace}hasMotor", motor_uri))
                triples.append((motor_uri, "rdf:type", f"{self.namespace}Motor"))
                triples.append((motor_uri, f"{self.namespace}motorSpecification", f'"{entry["Motor"]}"'))
            
            # Transmission
            if 'Getriebe' in entry:
                triples.append((subject, f"{self.namespace}hasTransmission", f'"{entry["Getriebe"]}"'))
            
            # Power
            if 'Leistung_kW' in entry:
                triples.append((subject, f"{self.namespace}powerOutput_kW", f'"{entry["Leistung_kW"]}"^^xsd:integer'))
            
            # Torque
            if 'Drehmoment_Nm' in entry:
                triples.append((subject, f"{self.namespace}torque_Nm", f'"{entry["Drehmoment_Nm"]}"^^xsd:integer'))
            
            # CO2 Emission
            if 'CO2_Emission' in entry:
                # Clean CO2 value
                co2 = entry['CO2_Emission'].strip()
                co2_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', co2)
                if co2_match:
                    triples.append((subject, f"{self.namespace}co2EmissionMin_g_km", f'"{co2_match.group(1)}"^^xsd:integer'))
                    triples.append((subject, f"{self.namespace}co2EmissionMax_g_km", f'"{co2_match.group(2)}"^^xsd:integer'))
        
        return triples
    
    def generate_turtle(self, triples: List[Tuple[str, str, str]]) -> str:
        """Generate Turtle (TTL) format for the triples"""
        turtle = []
        
        # Prefixes
        turtle.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        turtle.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
        turtle.append(f"@prefix {self.namespace} <http://example.com/bmw/> .")
        turtle.append("")
        
        # Group by subject for cleaner output
        subject_triples = {}
        for s, p, o in triples:
            if s not in subject_triples:
                subject_triples[s] = []
            subject_triples[s].append((p, o))
        
        # Generate Turtle
        for subject, predicates in subject_triples.items():
            turtle.append(f"{subject}")
            for i, (predicate, obj) in enumerate(predicates):
                if i == len(predicates) - 1:
                    turtle.append(f"    {predicate} {obj} .")
                else:
                    turtle.append(f"    {predicate} {obj} ;")
            turtle.append("")
        
        return "\n".join(turtle)
    
    def _clean_header(self, text: str) -> str:
        """Clean header text"""
        if not text:
            return ""
        # Remove line breaks and extra spaces
        cleaned = re.sub(r'\s+', ' ', str(text).strip())
        # Make valid property name
        cleaned = re.sub(r'[^\w]', '_', cleaned)
        return cleaned
    
    def _clean_value(self, text: str) -> str:
        """Clean cell value"""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', str(text).strip())