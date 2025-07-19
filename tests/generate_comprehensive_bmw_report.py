#!/usr/bin/env python3
"""
Generate comprehensive HTML reports for BMW documents with all segments,
context, and embedded visuals
"""

import asyncio
import json
import logging
import base64
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.pdf_processor import PDFProcessor
from core.parsers.interfaces.data_models import SegmentType, TextSubtype, Document
from core.parsers.utils.segment_context_enhancer import SegmentContextEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_table_content(content: str) -> str:
    """Format table content with SmolDocling tags into HTML"""
    if "<otsl>" in content:
        # SmolDocling table format
        lines = content.split("<nl>")
        if len(lines) > 1:
            html_table = "<table class='data-table'>"
            for i, line in enumerate(lines):
                if i == 0:  # Header row
                    html_table += "<tr>"
                    cells = line.replace("<otsl>", "").replace("<ched>", "|").split("|")
                    for cell in cells:
                        if cell.strip():
                            html_table += f"<th>{cell.strip()}</th>"
                    html_table += "</tr>"
                else:  # Data rows
                    html_table += "<tr>"
                    cells = line.replace("<fcel>", "|").split("|")
                    for cell in cells:
                        if cell.strip():
                            html_table += f"<td>{cell.strip()}</td>"
                    html_table += "</tr>"
            html_table += "</table>"
            return html_table
    
    # Fallback to simple table format
    if "|" in content and "\n" in content:
        lines = content.split("\n")
        html_table = "<table class='data-table'>"
        for i, line in enumerate(lines):
            if "|" in line:
                html_table += "<tr>"
                cells = line.split("|")
                tag = "th" if i == 0 else "td"
                for cell in cells:
                    if cell.strip():
                        html_table += f"<{tag}>{cell.strip()}</{tag}>"
                html_table += "</tr>"
        html_table += "</table>"
        return html_table
    
    return f"<pre class='table-content'>{content}</pre>"


def format_list_content(content: str) -> str:
    """Format list content into HTML"""
    lines = content.split("\n")
    html_list = "<ul class='formatted-list'>"
    
    for line in lines:
        line = line.strip()
        if line:
            # Remove bullet points and clean
            if line.startswith(("•", "●", "-", "*")):
                clean_line = line.lstrip("•●-* ").strip()
                html_list += f"<li>{clean_line}</li>"
            elif len(line) > 1 and line[0].isdigit() and line[1] in ".)":
                clean_line = line[2:].strip()
                html_list += f"<li>{clean_line}</li>"
            else:
                html_list += f"<li>{line}</li>"
    
    html_list += "</ul>"
    return html_list


def format_context(context: dict) -> str:
    """Format context information as HTML"""
    if not context:
        return "<p class='no-context'>No context available</p>"
    
    html = "<div class='context-info'>"
    html += "<h4>📍 Context Information:</h4>"
    html += "<ul>"
    
    if context.get("document_title"):
        html += f"<li><strong>Document:</strong> {context['document_title']}</li>"
    
    if context.get("document_type"):
        html += f"<li><strong>Type:</strong> {context['document_type']}</li>"
    
    if context.get("nearest_heading"):
        html += f"<li><strong>Section:</strong> {context['nearest_heading']}</li>"
    
    if context.get("table_reference"):
        html += f"<li><strong>Table Reference:</strong> {context['table_reference']}</li>"
    
    if context.get("list_introduction"):
        html += f"<li><strong>List Introduction:</strong> {context['list_introduction']}</li>"
    
    if context.get("position"):
        pos = context["position"]
        html += f"<li><strong>Position:</strong> Page {pos.get('page', 'N/A')}, {pos.get('relative_position', 'N/A')} of page</li>"
    
    html += "</ul></div>"
    return html


def generate_document_html(document: Document, pdf_name: str, output_path: Path):
    """Generate comprehensive HTML report for a single document"""
    
    # Group segments by page
    segments_by_page = {}
    for segment in document.segments:
        page = segment.page_number if segment.page_number is not None else 0
        if page not in segments_by_page:
            segments_by_page[page] = []
        segments_by_page[page].append(segment)
    
    # Start HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{pdf_name} - Comprehensive Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background: #1976d2;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .page-section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .page-header {{
            background: #e3f2fd;
            padding: 15px;
            margin: -20px -20px 20px -20px;
            border-radius: 8px 8px 0 0;
            font-size: 1.2em;
            font-weight: bold;
            color: #1976d2;
        }}
        .segment {{
            border: 1px solid #ddd;
            margin: 15px 0;
            padding: 15px;
            border-radius: 5px;
            background: #fafafa;
        }}
        .segment-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .segment-type {{
            background: #2196F3;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .segment-index {{
            color: #666;
            font-size: 0.9em;
        }}
        .context-info {{
            background: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
        .context-info h4 {{
            margin: 0 0 10px 0;
            color: #856404;
        }}
        .context-info ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .no-context {{
            color: #999;
            font-style: italic;
        }}
        .segment-content {{
            margin-top: 15px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .data-table th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .formatted-list {{
            background: #f0f0ff;
            padding: 15px 15px 15px 35px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
        }}
        .formatted-list li {{
            margin: 8px 0;
        }}
        .visual-element {{
            text-align: center;
            margin: 20px 0;
        }}
        .visual-element img {{
            max-width: 800px;
            max-height: 600px;
            border: 2px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .visual-description {{
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-radius: 5px;
            font-style: italic;
        }}
        pre {{
            background: #f5f5f5;
            padding: 10px;
            overflow-x: auto;
            border-radius: 3px;
            white-space: pre-wrap;
        }}
        .table-segment {{
            background: #e8f5e9;
        }}
        .list-segment {{
            background: #e3f2fd;
        }}
        .visual-segment {{
            background: #fce4ec;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #1976d2;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 {pdf_name}</h1>
        <p>Comprehensive Document Analysis with Context Enhancement</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="container">
        <div class="page-section">
            <h2>📊 Document Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{document.metadata.page_count}</div>
                    <div class="stat-label">Total Pages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(document.segments)}</div>
                    <div class="stat-label">Total Segments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len([s for s in document.segments if s.segment_type == SegmentType.TABLE])}</div>
                    <div class="stat-label">Tables</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len([s for s in document.segments if s.segment_subtype == TextSubtype.LIST.value or (s.segment_type == SegmentType.TEXT and any(p in s.content for p in ["•", "●", "- ", "* "]))])}</div>
                    <div class="stat-label">Lists</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(document.visual_elements)}</div>
                    <div class="stat-label">Visual Elements</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len([s for s in document.segments if "context" in s.metadata])}</div>
                    <div class="stat-label">Enhanced Segments</div>
                </div>
            </div>
        </div>
    """
    
    # Process each page
    for page_num in sorted(segments_by_page.keys()):
        page_segments = segments_by_page[page_num]
        
        html_content += f"""
        <div class="page-section">
            <div class="page-header">
                📄 Page {page_num + 1} - {len(page_segments)} segments
            </div>
        """
        
        # Process each segment on this page
        for segment in sorted(page_segments, key=lambda s: s.segment_index):
            # Determine segment CSS class
            segment_class = "segment"
            if segment.segment_type == SegmentType.TABLE:
                segment_class += " table-segment"
            elif segment.segment_type == SegmentType.VISUAL:
                segment_class += " visual-segment"
            elif segment.segment_subtype == TextSubtype.LIST.value or any(p in segment.content for p in ["•", "●", "- ", "* "]):
                segment_class += " list-segment"
            
            html_content += f"""
            <div class="{segment_class}">
                <div class="segment-header">
                    <div>
                        <span class="segment-type">{segment.segment_type.value if hasattr(segment.segment_type, 'value') else str(segment.segment_type)}</span>
                        {f'<span class="segment-subtype">({segment.segment_subtype})</span>' if segment.segment_subtype else ''}
                    </div>
                    <div class="segment-index">Segment #{segment.segment_index}</div>
                </div>
            """
            
            # Add context if available
            if "context" in segment.metadata:
                html_content += format_context(segment.metadata["context"])
            else:
                # Check if this segment should have context
                if SegmentContextEnhancer.needs_context(segment):
                    html_content += """
                    <div class="context-info" style="background: #ffebee; border-color: #f44336;">
                        <h4>⚠️ Context Missing</h4>
                        <p>This segment type should have context but none was found.</p>
                    </div>
                    """
            
            # Add segment content
            html_content += '<div class="segment-content">'
            
            if segment.segment_type == SegmentType.TABLE:
                html_content += "<h4>Table Content:</h4>"
                html_content += format_table_content(segment.content)
                
            elif segment.segment_type == SegmentType.VISUAL:
                # Try to find and embed the image
                if segment.visual_references:
                    visual_ref = segment.visual_references[0]
                    cache_dir = Path("cache/images")
                    
                    # Look for cached image
                    image_patterns = [
                        f"*{visual_ref}*.png",
                        f"*{visual_ref}*.jpg",
                        f"*{visual_ref}*.jpeg"
                    ]
                    
                    image_found = False
                    for pattern in image_patterns:
                        image_files = list(cache_dir.glob(pattern))
                        if image_files:
                            image_path = image_files[0]
                            try:
                                with open(image_path, "rb") as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode()
                                    html_content += f"""
                                    <div class="visual-element">
                                        <img src="data:image/png;base64,{img_data}" alt="Visual element">
                                        <div class="visual-description">{segment.content}</div>
                                    </div>
                                    """
                                    image_found = True
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to embed image: {e}")
                    
                    if not image_found:
                        html_content += f"""
                        <div class="visual-element">
                            <div style="padding: 40px; background: #f5f5f5; border: 2px dashed #ccc;">
                                <p>🖼️ Image not found in cache</p>
                                <p><small>Reference: {visual_ref}</small></p>
                            </div>
                            <div class="visual-description">{segment.content}</div>
                        </div>
                        """
                else:
                    html_content += f'<div class="visual-description">{segment.content}</div>'
                    
            elif segment.segment_subtype == TextSubtype.LIST.value or any(p in segment.content for p in ["•", "●", "- ", "* ", "1.", "2."]):
                html_content += "<h4>List Content:</h4>"
                html_content += format_list_content(segment.content)
                
            else:
                # Regular text content
                html_content += f"<pre>{segment.content}</pre>"
            
            # Add visual references if any
            if segment.visual_references and segment.segment_type != SegmentType.VISUAL:
                html_content += f'<p><small>🖼️ Visual references: {", ".join(segment.visual_references)}</small></p>'
            
            html_content += "</div></div>"
        
        html_content += "</div>"
    
    # Add visual elements summary at the end
    if document.visual_elements:
        html_content += """
        <div class="page-section">
            <h2>🖼️ All Visual Elements Summary</h2>
        """
        
        for i, ve in enumerate(document.visual_elements):
            html_content += f"""
            <div class="segment visual-segment">
                <div class="segment-header">
                    <div>
                        <span class="segment-type">{ve.element_type.value}</span>
                        <span>Page {ve.page_or_slide}</span>
                    </div>
                    <div class="segment-index">Visual Element #{i}</div>
                </div>
            """
            
            # Add visual element details
            if ve.analysis_metadata and "prompt" in ve.analysis_metadata:
                if "Context:" in ve.analysis_metadata["prompt"]:
                    html_content += """
                    <div class="context-info">
                        <h4>🔍 Analysis Context Used:</h4>
                        <pre style="font-size: 0.9em;">{}</pre>
                    </div>
                    """.format(ve.analysis_metadata["prompt"])
            
            html_content += f"""
                <div class="visual-description">
                    <p><strong>Description:</strong> {ve.vlm_description}</p>
                    <p><strong>Confidence:</strong> {ve.confidence:.2f}</p>
                    <p><strong>Hash:</strong> {ve.content_hash[:16]}...</p>
                </div>
            </div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </div>
</body>
</html>
    """
    
    output_path.write_text(html_content)
    logger.info(f"📄 Comprehensive report saved to: {output_path}")


async def process_all_bmw_documents():
    """Process all BMW documents and generate comprehensive reports"""
    
    pdf_files = [
        "Preview_BMW_X5_G05.pdf",
        "Preview_BMW_8er_G14_G15.pdf",
        "Preview_BMW_1er_Sedan_CN.pdf",
        "Preview_BMW_3er_G20.pdf"
    ]
    
    # Configure processor
    config = {
        "enable_preprocessing": True,
        "enable_image_analysis": True,
        "enable_page_analysis": False,
        "enable_context_enhancement": True,
        "parser_config": {
            "use_docling": True,
            "preserve_native_tags": True
        },
        "preprocessor_config": {
            "max_pages": 20  # Process all pages
        },
        "image_analyzer_config": {
            "max_images": 10  # Limit to 10 images per document for performance
        }
    }
    
    # Create processor
    processor = PDFProcessor(config)
    
    for pdf_name in pdf_files:
        pdf_path = Path("data/input") / pdf_name
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF not found: {pdf_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Processing: {pdf_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Process document
            document = await processor.parse_async(pdf_path)
            
            # Generate comprehensive HTML report
            output_path = Path(f"bmw_comprehensive_{pdf_name.replace('.pdf', '.html')}")
            generate_document_html(document, pdf_name, output_path)
            
            logger.info(f"✅ Successfully processed {pdf_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(process_all_bmw_documents())