#!/usr/bin/env python3
"""
Generate HTML report for Qwen2.5-VL testing with BMW documents

Creates a comprehensive HTML report showing:
- Page images
- VLM descriptions of pages
- All segments with content and type
- Embedded images with VLM content and JSON output
"""

import asyncio
import json
import base64
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import hashlib
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.hybrid_pdf_parser_qwen25 import HybridPDFParserQwen25
from core.parsers.interfaces import VisualElement, VisualElementType, DocumentType, SegmentType
from core.vlm.qwen25_processor import Qwen25VLMProcessor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to base64 for HTML embedding"""
    return base64.b64encode(image_data).decode('utf-8')


def render_table_content(content: str) -> Optional[str]:
    """Try to render table content as HTML table"""
    try:
        # Check if content contains table-like structure
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Remove table caption lines if present
        cleaned_lines = []
        for line in lines:
            if line.strip() and not (line.startswith('Table') and ':' in line):
                cleaned_lines.append(line)
        
        if len(cleaned_lines) < 2:
            return None
        
        lines = cleaned_lines
        
        # Try to detect delimiter
        delimiters = ['|', '\t', ',']
        delimiter = None
        for d in delimiters:
            if d in lines[0]:
                delimiter = d
                break
        
        # If no delimiter found, check if it's whitespace separated
        if not delimiter:
            # Check if lines have consistent spacing pattern (table-like)
            if any('BMW' in line for line in lines):
                # Parse as whitespace-separated table
                rows = []
                for line in lines:
                    if line.strip():
                        # Split by multiple spaces
                        cells = [cell.strip() for cell in line.split() if cell.strip()]
                        if cells:
                            rows.append(cells)
                
                if len(rows) >= 2:
                    # Build HTML table
                    html = '<table class="data-table">'
                    
                    # Header row
                    html += '<thead><tr>'
                    for cell in rows[0]:
                        html += f'<th>{cell}</th>'
                    html += '</tr></thead>'
                    
                    # Data rows
                    html += '<tbody>'
                    for row in rows[1:]:
                        html += '<tr>'
                        for i, cell in enumerate(row):
                            if i < len(rows[0]):
                                html += f'<td>{cell}</td>'
                        # Fill missing cells
                        for j in range(len(row), len(rows[0])):
                            html += '<td></td>'
                        html += '</tr>'
                    html += '</tbody>'
                    
                    html += '</table>'
                    return html
            
            return None
        
        # Parse table
        rows = []
        separator_line = -1
        
        for i, line in enumerate(lines):
            if line.strip():
                # Check if this is a markdown separator line
                if delimiter == '|' and all(c in '|- ' for c in line):
                    separator_line = i
                    continue
                    
                cells = [cell.strip() for cell in line.split(delimiter)]
                # Filter out empty cells at edges (common with | delimiter)
                if delimiter == '|':
                    cells = [c for c in cells if c]
                if cells:
                    rows.append(cells)
        
        if len(rows) < 2:
            return None
        
        # Build HTML table
        html = '<table class="data-table">'
        
        # Determine if first row is header (has separator after it in markdown)
        has_header = separator_line == 1
        
        if has_header:
            # Header row
            html += '<thead><tr>'
            for cell in rows[0]:
                html += f'<th>{cell}</th>'
            html += '</tr></thead>'
            
            # Data rows
            html += '<tbody>'
            for row in rows[1:]:
                html += '<tr>'
                for i, cell in enumerate(row):
                    # Ensure we don't exceed header columns
                    if i < len(rows[0]):
                        html += f'<td>{cell}</td>'
                # Fill missing cells
                for j in range(len(row), len(rows[0])):
                    html += '<td></td>'
                html += '</tr>'
            html += '</tbody>'
        else:
            # All rows in tbody
            html += '<tbody>'
            for row in rows:
                html += '<tr>'
                for cell in row:
                    html += f'<td>{cell}</td>'
                html += '</tr>'
            html += '</tbody>'
        
        html += '</table>'
        return html
        
    except Exception as e:
        logger.warning(f"Failed to render table: {e}")
        return None


def extract_page_images(pdf_path: Path, max_pages: int = 10) -> Dict[int, bytes]:
    """Extract page images from PDF"""
    page_images = {}
    
    doc = fitz.open(str(pdf_path))
    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        # Render at good quality
        mat = fitz.Matrix(2, 2)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)
        page_images[page_num + 1] = pix.tobytes("png")
    
    doc.close()
    return page_images


def extract_embedded_images_directly(pdf_path: Path, max_pages: int = 10) -> List[VisualElement]:
    """
    Extract embedded images directly from PDF using PyMuPDF
    Since SmolDocling doesn't extract them, we do it ourselves
    """
    visual_elements = []
    doc = fitz.open(str(pdf_path))
    
    for page_num in range(min(max_pages, len(doc))):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                # Convert to PNG if needed
                if pix.colorspace:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                img_data = pix.tobytes("png")
                
                # Get image position on page (if available)
                # For embedded images, we might not have exact position
                # so we use a placeholder
                bbox = [0, 0, pix.width, pix.height]
                
                # Create visual element
                ve = VisualElement(
                    element_type=VisualElementType.IMAGE,
                    source_format=DocumentType.PDF,
                    content_hash=hashlib.sha256(img_data).hexdigest()[:16],
                    page_or_slide=page_num + 1,
                    raw_data=img_data,
                    bounding_box=bbox,
                    analysis_metadata={
                        "source": "direct_extraction",
                        "original_size": [pix.width, pix.height],
                        "image_index": img_index
                    }
                )
                visual_elements.append(ve)
                logger.info(f"Extracted image {img_index} from page {page_num + 1}: {pix.width}x{pix.height}")
                
                pix = None  # Free memory
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
    
    doc.close()
    return visual_elements


def extract_embedded_images_with_loc_tags(pdf_path: Path, segments: List[Any]) -> List[VisualElement]:
    """
    Extract embedded images using loc tags from segments
    This is the approach we used before with PDFPlumber
    """
    visual_elements = []
    doc = fitz.open(str(pdf_path))
    
    for segment in segments:
        # Look for image/picture loc tags in segment content
        import re
        # Pattern for both image and picture tags: <image/picture><loc_x0><loc_y0><loc_x1><loc_y1>
        patterns = [
            r'<image><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>',
            r'<picture><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>',
            r'<img><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>',
            r'<figure><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        ]
        
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, segment.content))
        
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            page_num = segment.page_number
            
            if page_num <= len(doc):
                page = doc[page_num - 1]
                
                # Scale from SmolDocling coordinates (0-500) to page coordinates
                page_rect = page.rect
                scale_x = page_rect.width / 500.0
                scale_y = page_rect.height / 500.0
                
                # Convert coordinates
                rect_x0 = x0 * scale_x
                rect_y0 = y0 * scale_y
                rect_x1 = x1 * scale_x
                rect_y1 = y1 * scale_y
                
                # Extract image from region
                rect = fitz.Rect(rect_x0, rect_y0, rect_x1, rect_y1)
                mat = fitz.Matrix(2, 2)  # 2x zoom for quality
                pix = page.get_pixmap(matrix=mat, clip=rect)
                img_data = pix.tobytes("png")
                
                # Create visual element
                ve = VisualElement(
                    element_type=VisualElementType.IMAGE,
                    source_format=DocumentType.PDF,
                    content_hash=hashlib.sha256(img_data).hexdigest()[:16],
                    page_or_slide=page_num,
                    raw_data=img_data,
                    bounding_box=[x0, y0, x1, y1],
                    analysis_metadata={
                        "source": "loc_tag_extraction",
                        "original_size": [pix.width, pix.height]
                    }
                )
                visual_elements.append(ve)
                logger.info(f"Extracted embedded image from page {page_num} at {[x0, y0, x1, y1]}")
    
    doc.close()
    return visual_elements


def generate_html_report(
    pdf_path: Path,
    document: Any,
    page_images: Dict[int, bytes],
    page_contexts: Optional[Dict[int, Any]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Generate comprehensive HTML report"""
    
    if output_path is None:
        output_path = Path(f"bmw_vlm_report_{datetime.now():%Y%m%d_%H%M%S}.html")
    
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BMW VLM Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .page-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .page-image {
            max-width: 100%;
            border: 1px solid #ddd;
            margin: 10px 0;
        }
        .segment {
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        .segment-type {
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }
        .segment-content {
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background: #fff;
            padding: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            overflow-x: auto;
        }
        .visual-element {
            background: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #ffc107;
            border-radius: 4px;
        }
        .embedded-image {
            max-width: 400px;
            max-height: 400px;
            border: 1px solid #ddd;
            margin: 10px 0;
        }
        .vlm-description {
            background: #e7f3ff;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .json-output {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            overflow-x: auto;
        }
        .metadata {
            color: #666;
            font-size: 12px;
            margin-top: 5px;
        }
        .toc {
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toc a {
            color: #007bff;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .table-container {
            background: #e8f5e9;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #4caf50;
            border-radius: 4px;
            overflow-x: auto;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 10px 0;
        }
        .data-table th {
            background-color: #4caf50;
            color: white;
            padding: 8px;
            text-align: left;
            font-weight: bold;
        }
        .data-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .data-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .chart-container {
            background: #fff8e1;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #ff9800;
            border-radius: 4px;
        }
        .chart-label {
            font-weight: bold;
            color: #ff9800;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
"""]
    
    # Header
    html_parts.append(f"""
    <h1>BMW VLM Analysis Report</h1>
    <div class="metadata">
        <p><strong>Document:</strong> {pdf_path.name}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Pages:</strong> {document.metadata.page_count}</p>
        <p><strong>Total Segments:</strong> {len(document.segments)}</p>
        <p><strong>Visual Elements:</strong> {len(document.visual_elements)}</p>
    </div>
    """)
    
    # Table of Contents
    html_parts.append('<div class="toc"><h2>Table of Contents</h2><ul>')
    for page_num in sorted(page_images.keys()):
        html_parts.append(f'<li><a href="#page-{page_num}">Page {page_num}</a></li>')
    html_parts.append('</ul></div>')
    
    # Process each page
    for page_num in sorted(page_images.keys()):
        html_parts.append(f'<div class="page-section" id="page-{page_num}">')
        html_parts.append(f'<h2>Page {page_num}</h2>')
        
        # Page image
        img_data = page_images[page_num]
        img_base64 = image_to_base64(img_data)
        html_parts.append(f'<img class="page-image" src="data:image/png;base64,{img_base64}" alt="Page {page_num}">')
        
        # Page VLM description if available
        if page_contexts and page_num in page_contexts:
            context = page_contexts[page_num]
            html_parts.append('<div class="vlm-description">')
            html_parts.append('<h3>Page VLM Analysis</h3>')
            html_parts.append(f'<p><strong>Page Type:</strong> {context.page_type}</p>')
            if context.main_topic:
                html_parts.append(f'<p><strong>Main Topic:</strong> {context.main_topic}</p>')
            if context.key_information:
                html_parts.append('<p><strong>Key Information:</strong></p><ul>')
                for info in context.key_information:
                    html_parts.append(f'<li>{info}</li>')
                html_parts.append('</ul>')
            html_parts.append('</div>')
        
        # Segments for this page
        page_segments = [s for s in document.segments if s.page_number == page_num]
        if page_segments:
            html_parts.append('<h3>Segments</h3>')
            for i, segment in enumerate(page_segments):
                html_parts.append('<div class="segment">')
                html_parts.append(f'<div class="segment-type">Type: {segment.segment_type.value} / {segment.segment_subtype or "none"}</div>')
                html_parts.append(f'<div class="metadata">Index: {segment.segment_index}</div>')
                
                # Check if this is a table segment by type or content
                is_table = (segment.segment_type.value == 'table' or 
                           segment.segment_subtype == 'table' or
                           ('Table' in segment.content and 'BMW' in segment.content and 'Model' in segment.content) or
                           (segment.content.count('|') > 5) or  # Pipe-delimited table
                           ('BMW' in segment.content and any(model in segment.content for model in ['320i', '330i', 'M340i'])))
                
                if is_table:
                    html_parts.append('<div class="table-container">')
                    html_parts.append('<div class="chart-label">Table Content</div>')
                    
                    # Check if we have structured table data
                    if 'table_structure' in segment.metadata:
                        table_struct = segment.metadata['table_structure']
                        if 'headers' in table_struct and 'rows' in table_struct:
                            # Render structured table
                            html_parts.append('<table class="data-table">')
                            
                            # Headers
                            html_parts.append('<thead><tr>')
                            for header in table_struct['headers']:
                                html_parts.append(f'<th>{header}</th>')
                            html_parts.append('</tr></thead>')
                            
                            # Rows
                            html_parts.append('<tbody>')
                            for row in table_struct['rows']:
                                html_parts.append('<tr>')
                                for i, cell in enumerate(row):
                                    if i < len(table_struct['headers']):
                                        html_parts.append(f'<td>{cell}</td>')
                                html_parts.append('</tr>')
                            html_parts.append('</tbody>')
                            html_parts.append('</table>')
                            
                            # Show triple count
                            if 'triple_count' in segment.metadata:
                                html_parts.append(f'<div class="metadata">Extracted {segment.metadata["triple_count"]} knowledge graph triples</div>')
                    else:
                        # Try to parse and render table
                        table_html = render_table_content(segment.content)
                        if table_html:
                            html_parts.append(table_html)
                        else:
                            html_parts.append('<div class="segment-content">')
                            html_parts.append(segment.content)
                            html_parts.append('</div>')
                    
                    html_parts.append('</div>')
                else:
                    html_parts.append('<div class="segment-content">')
                    html_parts.append(segment.content)
                    html_parts.append('</div>')
                
                # Visual references
                if segment.visual_references:
                    html_parts.append(f'<div class="metadata">Visual References: {", ".join(segment.visual_references)}</div>')
                
                html_parts.append('</div>')
        
        # Visual elements for this page
        page_visuals = [ve for ve in document.visual_elements if ve.page_or_slide == page_num]
        if page_visuals:
            html_parts.append('<h3>Visual Elements</h3>')
            for ve in page_visuals:
                html_parts.append('<div class="visual-element">')
                html_parts.append(f'<h4>{ve.element_type.value}</h4>')
                
                # Embedded image
                if ve.raw_data:
                    img_base64 = image_to_base64(ve.raw_data)
                    html_parts.append(f'<img class="embedded-image" src="data:image/png;base64,{img_base64}" alt="{ve.element_type.value}">')
                
                # VLM Description
                if ve.vlm_description:
                    html_parts.append('<div class="vlm-description">')
                    html_parts.append('<strong>VLM Description:</strong>')
                    html_parts.append(f'<p>{ve.vlm_description}</p>')
                    if ve.confidence_score:
                        html_parts.append(f'<p class="metadata">Confidence: {ve.confidence_score:.2%}</p>')
                    html_parts.append('</div>')
                
                # Structured data / JSON output
                if hasattr(ve, 'analysis_metadata') and ve.analysis_metadata.get('structured_data'):
                    html_parts.append('<div>')
                    html_parts.append('<strong>Structured Data (JSON):</strong>')
                    html_parts.append('<div class="json-output">')
                    html_parts.append(json.dumps(ve.analysis_metadata['structured_data'], indent=2, ensure_ascii=False))
                    html_parts.append('</div>')
                    html_parts.append('</div>')
                
                # Metadata
                html_parts.append('<div class="metadata">')
                html_parts.append(f'<p>Content Hash: {ve.content_hash}</p>')
                if ve.bounding_box:
                    html_parts.append(f'<p>Bounding Box: {ve.bounding_box}</p>')
                if ve.analysis_metadata:
                    html_parts.append(f'<p>Source: {ve.analysis_metadata.get("source", "unknown")}</p>')
                html_parts.append('</div>')
                
                html_parts.append('</div>')
        
        html_parts.append('</div>')  # End page section
    
    html_parts.append('</body></html>')
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    return output_path


async def test_bmw_with_html_report():
    """Test BMW documents and generate HTML report"""
    
    # Find BMW PDF files
    input_dir = Path("data/input")
    bmw_files = list(input_dir.glob("*BMW*.pdf")) + list(input_dir.glob("*bmw*.pdf"))
    
    if not bmw_files:
        logger.error("No BMW PDF files found in data/input")
        return
    
    # Use first BMW file found
    pdf_path = bmw_files[0]
    logger.info(f"Testing with: {pdf_path.name}")
    
    # Initialize parser with VLM
    parser = HybridPDFParserQwen25(
        config={
            "max_pages": 10,  # Limit for testing
            "pdfplumber_mode": 1,
            "enable_page_context": True,
            "page_context_pages": 5,
            "environment": "development",  # Ensure docling is enabled
            "vlm": {
                "temperature": 0.2,
                "max_new_tokens": 512,
                "batch_size": 2,
                "enable_structured_parsing": True
            },
            "image_extraction": {
                "min_size": 100,
                "extract_embedded": True,
                "render_fallback": True
            }
        },
        enable_vlm=True
    )
    
    # Parse document
    logger.info("Parsing document...")
    document = await parser.parse(pdf_path)
    
    # Extract page images for report
    logger.info("Extracting page images...")
    page_images = extract_page_images(pdf_path, max_pages=10)
    
    # If no visual elements found, try direct extraction
    if len(document.visual_elements) == 0:
        logger.info("No visual elements found by parser, extracting images directly...")
        # First try loc tag extraction
        embedded_visuals = extract_embedded_images_with_loc_tags(pdf_path, document.segments)
        
        if not embedded_visuals:
            # If no loc tags, extract all embedded images directly
            logger.info("No loc tags found, extracting all embedded images directly from PDF...")
            embedded_visuals = extract_embedded_images_directly(pdf_path, max_pages=10)
        
        if embedded_visuals:
            logger.info(f"Found {len(embedded_visuals)} embedded images")
            
            # Process with VLM
            vlm_processor = parser.vlm_processor
            if vlm_processor:
                logger.info("Processing embedded images with VLM...")
                results = await vlm_processor.process_visual_elements(embedded_visuals)
                
                # Update visual elements with results
                for ve, result in zip(embedded_visuals, results):
                    if result.success:
                        ve.vlm_description = result.description
                        ve.confidence_score = result.confidence
                        if result.structured_data:
                            ve.analysis_metadata['structured_data'] = result.structured_data
                
                # Add to document
                document.visual_elements.extend(embedded_visuals)
    
    # Get page contexts if available
    page_contexts = None
    if parser.enable_page_context and hasattr(parser, '_page_contexts'):
        page_contexts = parser._page_contexts
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    report_path = generate_html_report(
        pdf_path=pdf_path,
        document=document,
        page_images=page_images,
        page_contexts=page_contexts,
        output_path=Path(f"bmw_vlm_report_{pdf_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.html")
    )
    
    logger.info(f"✅ HTML report generated: {report_path}")
    logger.info(f"   Total pages: {len(page_images)}")
    logger.info(f"   Total segments: {len(document.segments)}")
    logger.info(f"   Visual elements: {len(document.visual_elements)}")
    
    # Cleanup
    parser.cleanup()
    
    return report_path


if __name__ == "__main__":
    report_path = asyncio.run(test_bmw_with_html_report())
    if report_path:
        print(f"\n✅ Report generated successfully: {report_path}")
        print(f"Open the HTML file in your browser to view the results.")
    else:
        print("\n❌ Report generation failed!")