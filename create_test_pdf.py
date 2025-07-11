#!/usr/bin/env python3
"""
Create a simple test PDF for SmolDocling testing
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_test_pdf():
    """Create a simple test PDF with clear text"""
    pdf_path = "data/input/test_simple.pdf"
    
    # Create PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Test Document")
    
    # Add some paragraphs
    c.setFont("Helvetica", 12)
    y_position = height - 2*inch
    
    lines = [
        "This is a simple test document.",
        "It contains clear text that should be easy to read.",
        "",
        "Here is a list:",
        "- Item 1",
        "- Item 2", 
        "- Item 3",
        "",
        "This is the end of the document."
    ]
    
    for line in lines:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.3*inch
    
    # Save PDF
    c.save()
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_test_pdf()