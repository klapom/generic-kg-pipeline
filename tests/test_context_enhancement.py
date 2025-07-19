#!/usr/bin/env python3
"""
Test context enhancement for tables and lists
Verifies that SegmentContextEnhancer properly adds context to segments
and that ImageAnalyzer uses this context in VLM prompts
"""

import asyncio
import json
import logging
import sys
import base64
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.pdf_processor import PDFProcessor
from core.parsers.interfaces.data_models import Segment, SegmentType, TextSubtype
from core.parsers.utils.segment_context_enhancer import SegmentContextEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_context_enhancement(segments: List[Segment]) -> dict:
    """Analyze which segments received context enhancement"""
    stats = {
        "total_segments": len(segments),
        "enhanced_segments": 0,
        "tables_with_context": 0,
        "lists_with_context": 0,
        "visuals_with_context": 0,
        "context_details": [],
        "all_tables": 0,
        "all_lists": 0,
        "all_visuals": 0
    }
    
    for segment in segments:
        # Count all segment types
        if segment.segment_type == SegmentType.TABLE:
            stats["all_tables"] += 1
        elif segment.segment_subtype == TextSubtype.LIST.value or ("•" in segment.content and segment.segment_type == SegmentType.TEXT):
            stats["all_lists"] += 1
        elif segment.segment_type == SegmentType.VISUAL:
            stats["all_visuals"] += 1
            
        if "context" in segment.metadata:
            stats["enhanced_segments"] += 1
            
            context_info = {
                "segment_index": segment.segment_index,
                "segment_type": segment.segment_type,
                "segment_subtype": segment.segment_subtype,
                "page": segment.page_number,
                "context": segment.metadata["context"]
            }
            
            # Count by type
            if segment.segment_type == SegmentType.TABLE:
                stats["tables_with_context"] += 1
                context_info["content_preview"] = segment.content[:100] + "..."
            elif segment.segment_subtype == TextSubtype.LIST.value or ("•" in segment.content and segment.segment_type == SegmentType.TEXT):
                stats["lists_with_context"] += 1
                context_info["content_preview"] = segment.content[:100] + "..."
            elif segment.segment_type == SegmentType.VISUAL:
                stats["visuals_with_context"] += 1
                context_info["visual_refs"] = segment.visual_references
            
            stats["context_details"].append(context_info)
    
    return stats


def generate_html_report(document, context_stats: dict, output_path: Path):
    """Generate HTML report showing context enhancement results"""
    # Extract SmolDocling table content if available
    def format_table_content(content: str) -> str:
        """Format table content with SmolDocling tags into HTML"""
        if "<otsl>" in content:
            # SmolDocling table format
            lines = content.split("<nl>")
            if len(lines) > 1:
                html_table = "<table border='1' style='border-collapse: collapse; margin: 10px 0;'>"
                for i, line in enumerate(lines):
                    if i == 0:  # Header row
                        html_table += "<tr>"
                        cells = line.replace("<otsl>", "").replace("<ched>", "|").split("|")
                        for cell in cells:
                            if cell.strip():
                                html_table += f"<th style='padding: 5px; background: #f0f0f0;'>{cell.strip()}</th>"
                        html_table += "</tr>"
                    else:  # Data rows
                        html_table += "<tr>"
                        cells = line.replace("<fcel>", "|").split("|")
                        for cell in cells:
                            if cell.strip():
                                html_table += f"<td style='padding: 5px;'>{cell.strip()}</td>"
                        html_table += "</tr>"
                html_table += "</table>"
                return html_table
        
        # Fallback to simple table format
        if "|" in content:
            lines = content.split("\n")
            html_table = "<table border='1' style='border-collapse: collapse; margin: 10px 0;'>"
            for i, line in enumerate(lines):
                if "|" in line:
                    html_table += "<tr>"
                    cells = line.split("|")
                    tag = "th" if i == 0 else "td"
                    style = "padding: 5px; background: #f0f0f0;" if i == 0 else "padding: 5px;"
                    for cell in cells:
                        if cell.strip():
                            html_table += f"<{tag} style='{style}'>{cell.strip()}</{tag}>"
                    html_table += "</tr>"
            html_table += "</table>"
            return html_table
        
        return f"<pre>{content}</pre>"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Context Enhancement Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .stats {{
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
            color: #2196F3;
        }}
        .segment-detail {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .context-info {{
            background: #e8f5e9;
            padding: 10px;
            margin: 10px 0;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        pre {{
            background: #f5f5f5;
            padding: 10px;
            overflow-x: auto;
            border-radius: 3px;
        }}
        .visual-element {{
            border: 2px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📋 Context Enhancement Test Report</h1>
        <p>Document: {document.metadata.title or document.source_path.name}</p>
        <p>Generated: {document.metadata.created_date}</p>
        
        <h2>📊 Enhancement Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{context_stats['total_segments']}</div>
                <div>Total Segments</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{context_stats['enhanced_segments']}</div>
                <div>Enhanced Segments</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{context_stats.get('all_tables', 0)} / {context_stats['tables_with_context']}</div>
                <div>Tables (Total / Enhanced)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{context_stats.get('all_lists', 0)} / {context_stats['lists_with_context']}</div>
                <div>Lists (Total / Enhanced)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{context_stats.get('all_visuals', 0)} / {context_stats['visuals_with_context']}</div>
                <div>Visuals (Total / Enhanced)</div>
            </div>
        </div>
        
        <h2>🔍 Context Enhancement Details</h2>
    """
    
    # Add details for each enhanced segment
    for detail in context_stats["context_details"]:
        context_parts = []
        context = detail["context"]
        
        if context.get("nearest_heading"):
            context_parts.append(f"<strong>Heading:</strong> {context['nearest_heading']}")
        if context.get("document_type"):
            context_parts.append(f"<strong>Doc Type:</strong> {context['document_type']}")
        if context.get("table_reference"):
            context_parts.append(f"<strong>Table Ref:</strong> {context['table_reference']}")
        if context.get("list_introduction"):
            context_parts.append(f"<strong>List Intro:</strong> {context['list_introduction']}")
        
        html_content += f"""
        <div class="segment-detail">
            <h3>Segment #{detail['segment_index']} - {detail['segment_type']} 
                {f"({detail['segment_subtype']})" if detail['segment_subtype'] else ""}</h3>
            <p><strong>Page:</strong> {detail['page']}</p>
            <div class="context-info">
                {"<br>".join(context_parts)}
            </div>
            {f"<pre>{detail.get('content_preview', '')}</pre>" if detail.get('content_preview') else ""}
        </div>
        """
    
    # Add ALL segments with their content and context
    html_content += """
        <h2>📝 All Document Segments</h2>
    """
    
    for segment in document.segments:
        segment_style = "background: #f9f9f9; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;"
        
        # Highlight segments with context
        if "context" in segment.metadata:
            segment_style = "background: #e8f5e9; border: 2px solid #4CAF50; padding: 15px; margin: 10px 0; border-radius: 5px;"
        
        html_content += f"""
        <div style="{segment_style}">
            <h3>Segment #{segment.segment_index} - {segment.segment_type} 
                {f"({segment.segment_subtype})" if segment.segment_subtype else ""}</h3>
            <p><strong>Page:</strong> {segment.page_number}</p>
        """
        
        # Add context information if available
        if "context" in segment.metadata:
            context = segment.metadata["context"]
            context_parts = []
            
            if context.get("nearest_heading"):
                context_parts.append(f"<strong>📍 Nearest Heading:</strong> {context['nearest_heading']}")
            if context.get("document_type"):
                context_parts.append(f"<strong>📄 Document Type:</strong> {context['document_type']}")
            if context.get("table_reference"):
                context_parts.append(f"<strong>📊 Table Reference:</strong> {context['table_reference']}")
            if context.get("list_introduction"):
                context_parts.append(f"<strong>📋 List Introduction:</strong> {context['list_introduction']}")
            if context.get("position"):
                pos = context["position"]
                context_parts.append(f"<strong>📐 Position:</strong> {pos['relative_position']} of page")
            
            if context_parts:
                html_content += """
                <div style="background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 3px; border-left: 4px solid #ffc107;">
                    <h4 style="margin-top: 0;">🔍 Context Information:</h4>
                """ + "<br>".join(context_parts) + """
                </div>
                """
        
        # Show why context is missing if not enhanced
        if "context" not in segment.metadata:
            # Check if this segment type should have context
            if SegmentContextEnhancer.needs_context(segment):
                html_content += """
                <div style="background: #ffebee; padding: 10px; margin: 10px 0; border-radius: 3px; border-left: 4px solid #f44336;">
                    <h4 style="margin-top: 0; color: #d32f2f;">⚠️ Context Missing</h4>
                    <p style="color: #d32f2f;">This segment type should have context but none was added.</p>
                    <small>Segment needs context: YES | Context found: NO</small>
                </div>
                """
            else:
                html_content += """
                <div style="background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 3px; border-left: 4px solid #2196F3;">
                    <h4 style="margin-top: 0; color: #1976d2;">ℹ️ No Context Needed</h4>
                    <p style="color: #1976d2;">This segment type does not require context enhancement.</p>
                    <small>Regular text/paragraph segments don't need context</small>
                </div>
                """
        
        # Add segment content
        html_content += "<div style='margin-top: 10px;'><strong>Content:</strong></div>"
        
        if segment.segment_type == SegmentType.TABLE:
            html_content += format_table_content(segment.content)
        elif segment.segment_subtype == TextSubtype.LIST.value or "•" in segment.content or "●" in segment.content:
            # Format list content
            list_items = segment.content.split("\n")
            html_content += "<ul style='margin: 10px 0; background: #f0f0ff; padding: 15px; border-left: 4px solid #2196F3;'>"
            for item in list_items:
                item = item.strip()
                if item and (item.startswith("•") or item.startswith("●") or item.startswith("-")):
                    # Remove bullet and clean
                    clean_item = item.lstrip("•●- ").strip()
                    html_content += f"<li style='margin: 5px 0;'>{clean_item}</li>"
                elif item:
                    html_content += f"<li style='margin: 5px 0;'>{item}</li>"
            html_content += "</ul>"
        else:
            # Regular content
            html_content += f"<pre style='background: #f5f5f5; padding: 10px; overflow-x: auto;'>{segment.content}</pre>"
        
        # Add visual references if any
        if segment.visual_references:
            html_content += f"<p><small>🖼️ Visual references: {', '.join(segment.visual_references)}</small></p>"
        
        html_content += "</div>"
    
    # Add visual elements with enhanced prompts
    html_content += """
        <h2>🖼️ Visual Elements with Context</h2>
    """
    
    # Get image cache directory
    cache_dir = Path("cache/images")
    
    for ve in document.visual_elements:
        html_content += f"""
        <div class="visual-element">
            <h3>{ve.element_type.value} - Page {ve.page_or_slide}</h3>
        """
        
        # Try to find and embed the image
        if ve.content_hash:
            # Look for image in cache
            possible_patterns = [
                f"*{ve.content_hash}*.png",
                f"*{ve.content_hash}*.jpg",
                f"*{ve.content_hash}*.jpeg"
            ]
            
            image_found = False
            for pattern in possible_patterns:
                image_files = list(cache_dir.glob(pattern))
                if image_files:
                    image_path = image_files[0]
                    try:
                        import base64
                        with open(image_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            html_content += f"""
                            <div style="text-align: center; margin: 15px 0;">
                                <img src="data:image/png;base64,{img_data}" 
                                     style="max-width: 600px; max-height: 400px; border: 1px solid #ddd; border-radius: 4px;"
                                     alt="{ve.element_type.value}">
                            </div>
                            """
                            image_found = True
                            break
                    except Exception as e:
                        logger.warning(f"Failed to embed image: {e}")
            
            if not image_found:
                html_content += f"""
                <div style="text-align: center; margin: 15px 0; padding: 20px; background: #f5f5f5; border: 1px dashed #ccc; border-radius: 4px;">
                    <p style="color: #666;">🖼️ Image not found in cache (hash: {ve.content_hash[:8]}...)</p>
                </div>
                """
        
        html_content += f"""
            <p><strong>Description:</strong> {ve.vlm_description}</p>
        """
        
        # Show enhanced prompt if context was used
        if ve.analysis_metadata and "prompt" in ve.analysis_metadata:
            prompt = ve.analysis_metadata.get('prompt', 'No prompt recorded')
            if "Context:" in prompt:
                html_content += f"""
                <div class="context-info">
                    <strong>🔍 Enhanced Analysis Prompt (with context):</strong><br>
                    <pre style="background: #f5f5f5; padding: 10px; font-size: 0.9em; overflow-x: auto;">{prompt}</pre>
                </div>
                """
        
        # Show extracted data if available
        if ve.extracted_data:
            html_content += f"""
            <div style="background: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 3px;">
                <strong>📊 Extracted Data:</strong>
                <pre>{json.dumps(ve.extracted_data, indent=2)}</pre>
            </div>
            """
        
        html_content += f"""
            <p><strong>Confidence:</strong> {ve.confidence:.2f}</p>
            <p><small>Type: {ve.element_type.value} | Hash: {ve.content_hash[:16]}...</small></p>
        </div>
        """
    
    html_content += """
    </div>
</body>
</html>
    """
    
    output_path.write_text(html_content)
    logger.info(f"📄 HTML report saved to: {output_path}")


async def test_context_enhancement():
    """Test context enhancement with a BMW PDF"""
    # Get test PDF
    test_dir = Path("data/input")
    
    # Use the comprehensive test PDF with tables, lists and images
    test_pdf = test_dir / "bmw_comprehensive_test.pdf"
    if not test_pdf.exists():
        # Fallback to real BMW PDFs
        test_pdf = test_dir / "Preview_BMW_3er_G20.pdf"
        if not test_pdf.exists():
            test_pdf = test_dir / "Preview_BMW_1er_Sedan_CN.pdf"
            if not test_pdf.exists():
                logger.error("❌ No BMW PDF files found in data/input")
                return
    
    logger.info(f"📚 Testing with: {test_pdf}")
    
    # Configure processor with context enhancement
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
            "max_pages": 5  # Limit to first 5 pages for testing
        },
        "image_analyzer_config": {
            "max_images": 10  # Limit number of images to analyze
        }
    }
    
    # Create processor
    processor = PDFProcessor(config)
    
    # Process document
    logger.info("🔄 Processing document with context enhancement...")
    document = await processor.parse_async(test_pdf)
    
    # Analyze context enhancement
    context_stats = analyze_context_enhancement(document.segments)
    
    # Log summary
    logger.info(f"""
✅ Context Enhancement Results:
- Total segments: {context_stats['total_segments']}
- Enhanced segments: {context_stats['enhanced_segments']}
- Tables with context: {context_stats['tables_with_context']}
- Lists with context: {context_stats['lists_with_context']}
- Visuals with context: {context_stats['visuals_with_context']}
    """)
    
    # Log some examples
    if context_stats["context_details"]:
        logger.info("📋 Example enhanced segments:")
        for detail in context_stats["context_details"][:3]:
            logger.info(f"  - Segment #{detail['segment_index']}: {detail['segment_type']}")
            if detail["context"].get("nearest_heading"):
                logger.info(f"    Heading: {detail['context']['nearest_heading']}")
            if detail["context"].get("table_reference"):
                logger.info(f"    Table ref: {detail['context']['table_reference']}")
    
    # Generate HTML report
    output_path = Path("context_enhancement_test_report.html")
    generate_html_report(document, context_stats, output_path)
    
    # Save detailed JSON for debugging
    json_path = Path("context_enhancement_details.json")
    with open(json_path, "w") as f:
        json.dump({
            "document_title": document.metadata.title,
            "total_segments": len(document.segments),
            "total_visuals": len(document.visual_elements),
            "context_stats": context_stats
        }, f, indent=2)
    logger.info(f"📊 Detailed stats saved to: {json_path}")


if __name__ == "__main__":
    asyncio.run(test_context_enhancement())