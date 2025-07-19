#!/usr/bin/env python3
"""
Create a test PDF with tables and charts for VLM testing
"""

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import numpy as np
from pathlib import Path
import tempfile


def create_test_pdf_with_tables():
    """Create a test PDF containing tables and charts"""
    
    # Create output path
    output_path = Path("data/input/test_tables_charts.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create document
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("Test Document: Tables and Charts for VLM Analysis", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.5*inch))
    
    # Create a simple data table
    subtitle1 = Paragraph("Table 1: BMW Model Specifications", styles['Heading2'])
    story.append(subtitle1)
    story.append(Spacer(1, 0.2*inch))
    
    data = [
        ['Model', 'Engine', 'Power (HP)', 'Torque (Nm)', 'Price (EUR)'],
        ['BMW 320i', '2.0L Turbo', '184', '300', '45,900'],
        ['BMW 330i', '2.0L Turbo', '258', '400', '52,300'],
        ['BMW M340i', '3.0L Turbo', '387', '500', '68,500'],
        ['BMW 320d', '2.0L Diesel', '190', '400', '47,200'],
        ['BMW 330e', '2.0L Hybrid', '292', '420', '54,800']
    ]
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*inch))
    
    # Create a bar chart
    subtitle2 = Paragraph("Chart 1: Power Comparison", styles['Heading2'])
    story.append(subtitle2)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate bar chart
    models = ['320i', '330i', 'M340i', '320d', '330e']
    power = [184, 258, 387, 190, 292]
    
    plt.figure(figsize=(8, 6))
    plt.bar(models, power, color=['#0066CC', '#0066CC', '#FF6600', '#00AA44', '#9933FF'])
    plt.title('BMW 3 Series Power Output Comparison')
    plt.xlabel('Model')
    plt.ylabel('Power (HP)')
    plt.grid(axis='y', alpha=0.3)
    
    # Save chart
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        chart1_path = tmp.name
    plt.close()
    
    # Add chart to PDF
    img1 = Image(chart1_path, width=5*inch, height=3.75*inch)
    story.append(img1)
    story.append(Spacer(1, 0.5*inch))
    
    # Create another table with different data
    subtitle3 = Paragraph("Table 2: Fuel Efficiency Comparison", styles['Heading2'])
    story.append(subtitle3)
    story.append(Spacer(1, 0.2*inch))
    
    efficiency_data = [
        ['Model', 'City (L/100km)', 'Highway (L/100km)', 'Combined (L/100km)', 'CO2 (g/km)'],
        ['BMW 320i', '7.8', '5.4', '6.4', '146'],
        ['BMW 330i', '8.2', '5.7', '6.7', '153'],
        ['BMW M340i', '10.1', '6.8', '8.1', '185'],
        ['BMW 320d', '5.2', '3.9', '4.4', '116'],
        ['BMW 330e', '1.8', '1.5', '1.6', '37']
    ]
    
    t2 = Table(efficiency_data)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*inch))
    
    # Create a pie chart
    subtitle4 = Paragraph("Chart 2: Market Share Distribution", styles['Heading2'])
    story.append(subtitle4)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate pie chart
    sizes = [35, 25, 20, 15, 5]
    labels = ['320i', '330i', 'M340i', '320d', '330e']
    colors_pie = ['#0066CC', '#00AA44', '#FF6600', '#FFD700', '#9933FF']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.title('BMW 3 Series Sales Distribution 2024')
    
    # Save pie chart
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        chart2_path = tmp.name
    plt.close()
    
    # Add pie chart to PDF
    img2 = Image(chart2_path, width=4*inch, height=4*inch)
    story.append(img2)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temp files
    Path(chart1_path).unlink()
    Path(chart2_path).unlink()
    
    print(f"Test PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_test_pdf_with_tables()