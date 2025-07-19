#!/usr/bin/env python3
"""
Process multiple BMW PDF documents with context enhancement
"""

import asyncio
import json
import logging
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.pdf_processor import PDFProcessor
from core.parsers.interfaces.data_models import SegmentType, TextSubtype

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_document(document, pdf_name):
    """Analyze a processed document and return statistics"""
    stats = {
        "pdf_name": pdf_name,
        "title": document.metadata.title or "(anonymous)",
        "total_segments": len(document.segments),
        "total_pages": document.metadata.page_count,
        "visual_elements": len(document.visual_elements),
        "tables": 0,
        "lists": 0,
        "headings": 0,
        "paragraphs": 0,
        "enhanced_segments": 0,
        "processing_time": 0  # Will be calculated separately
    }
    
    # Count segment types
    for segment in document.segments:
        if segment.segment_type == SegmentType.TABLE:
            stats["tables"] += 1
        elif segment.segment_subtype == TextSubtype.LIST.value or (
            segment.segment_type == SegmentType.TEXT and 
            any(pattern in segment.content for pattern in ["•", "●", "- ", "* "])
        ):
            stats["lists"] += 1
        elif segment.segment_subtype in [TextSubtype.HEADING_1.value, TextSubtype.HEADING_2.value, 
                                        TextSubtype.HEADING_3.value, TextSubtype.TITLE.value]:
            stats["headings"] += 1
        elif segment.segment_type == SegmentType.TEXT:
            stats["paragraphs"] += 1
            
        # Check if enhanced
        if "context" in segment.metadata:
            stats["enhanced_segments"] += 1
    
    return stats


def generate_summary_html(all_stats, output_path):
    """Generate summary HTML report for all processed documents"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW Documents Processing Summary</title>
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
        h1 {{
            color: #333;
            text-align: center;
        }}
        .doc-summary {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .doc-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .stat-item {{
            background: #e3f2fd;
            padding: 10px;
            border-radius: 3px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1976d2;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
        }}
        .processing-info {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f0f0f0;
            font-weight: bold;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 BMW Documents Processing Summary</h1>
        <p style="text-align: center; color: #666;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>📊 Overview</h2>
        <table class="summary-table">
            <tr>
                <th>Document</th>
                <th>Pages</th>
                <th>Segments</th>
                <th>Tables</th>
                <th>Lists</th>
                <th>Images</th>
                <th>Enhanced</th>
                <th>Time (s)</th>
            </tr>
    """
    
    # Add summary row for each document
    total_pages = 0
    total_segments = 0
    total_tables = 0
    total_lists = 0
    total_images = 0
    total_enhanced = 0
    total_time = 0
    
    for stats in all_stats:
        html_content += f"""
            <tr>
                <td>{stats['pdf_name']}</td>
                <td>{stats['total_pages']}</td>
                <td>{stats['total_segments']}</td>
                <td>{stats['tables']}</td>
                <td>{stats['lists']}</td>
                <td>{stats['visual_elements']}</td>
                <td>{stats['enhanced_segments']}</td>
                <td>{stats['processing_time']:.1f}</td>
            </tr>
        """
        total_pages += stats['total_pages']
        total_segments += stats['total_segments']
        total_tables += stats['tables']
        total_lists += stats['lists']
        total_images += stats['visual_elements']
        total_enhanced += stats['enhanced_segments']
        total_time += stats['processing_time']
    
    # Add totals row
    html_content += f"""
            <tr style="font-weight: bold; background-color: #e8f5e9;">
                <td>TOTAL</td>
                <td>{total_pages}</td>
                <td>{total_segments}</td>
                <td>{total_tables}</td>
                <td>{total_lists}</td>
                <td>{total_images}</td>
                <td>{total_enhanced}</td>
                <td>{total_time:.1f}</td>
            </tr>
        </table>
        
        <h2>📋 Detailed Results</h2>
    """
    
    # Add detailed section for each document
    for stats in all_stats:
        html_content += f"""
        <div class="doc-summary">
            <div class="doc-title">{stats['pdf_name']}</div>
            <p><strong>Title:</strong> {stats['title']}</p>
            
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{stats['total_pages']}</div>
                    <div class="stat-label">Pages</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['total_segments']}</div>
                    <div class="stat-label">Total Segments</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['tables']}</div>
                    <div class="stat-label">Tables</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['lists']}</div>
                    <div class="stat-label">Lists</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['headings']}</div>
                    <div class="stat-label">Headings</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['paragraphs']}</div>
                    <div class="stat-label">Paragraphs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['visual_elements']}</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{stats['enhanced_segments']}</div>
                    <div class="stat-label">Enhanced</div>
                </div>
            </div>
            
            <div class="processing-info">
                ⏱️ Processing time: {stats['processing_time']:.2f} seconds
            </div>
        </div>
        """
    
    html_content += """
    </div>
</body>
</html>
    """
    
    output_path.write_text(html_content)
    logger.info(f"📄 Summary HTML report saved to: {output_path}")


async def process_bmw_documents():
    """Process all BMW documents"""
    # Define PDFs to process
    pdf_files = [
        "Preview_BMW_X5_G05.pdf",
        "Preview_BMW_8er_G14_G15.pdf", 
        "Preview_BMW_1er_Sedan_CN.pdf",
        "Preview_BMW_3er_G20.pdf"  # Also include 3er for comparison
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
            "max_pages": 10  # Process up to 10 pages per document
        },
        "image_analyzer_config": {
            "max_images": 20  # Analyze up to 20 images per document
        }
    }
    
    # Create processor
    processor = PDFProcessor(config)
    
    # Process each PDF
    all_stats = []
    
    for pdf_name in pdf_files:
        pdf_path = Path("data/input") / pdf_name
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF not found: {pdf_path}")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Processing: {pdf_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Process document with timing
            import time
            start_time = time.time()
            document = await processor.parse_async(pdf_path)
            processing_time = time.time() - start_time
            
            # Analyze results
            stats = analyze_document(document, pdf_name)
            stats["processing_time"] = processing_time
            all_stats.append(stats)
            
            # Log summary
            logger.info(f"""
✅ Processed {pdf_name}:
   - Pages: {stats['total_pages']}
   - Segments: {stats['total_segments']}
   - Tables: {stats['tables']}
   - Lists: {stats['lists']}
   - Images: {stats['visual_elements']}
   - Enhanced segments: {stats['enhanced_segments']}
   - Processing time: {stats['processing_time']:.2f}s
            """)
            
            # Save individual JSON results
            json_path = Path(f"bmw_results_{pdf_name.replace('.pdf', '.json')}")
            with open(json_path, "w") as f:
                json.dump({
                    "stats": stats,
                    "segments_sample": [
                        {
                            "index": seg.segment_index,
                            "type": seg.segment_type.value if hasattr(seg.segment_type, 'value') else str(seg.segment_type),
                            "subtype": seg.segment_subtype,
                            "page": seg.page_number,
                            "has_context": "context" in seg.metadata,
                            "content_preview": seg.content[:100] + "..." if len(seg.content) > 100 else seg.content
                        }
                        for seg in document.segments[:10]  # First 10 segments as sample
                    ]
                }, f, indent=2)
            logger.info(f"📊 Detailed results saved to: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    if all_stats:
        summary_path = Path("bmw_documents_summary.html")
        generate_summary_html(all_stats, summary_path)
        
        # Save all stats as JSON
        all_stats_path = Path("bmw_documents_all_stats.json")
        with open(all_stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"📊 All statistics saved to: {all_stats_path}")


if __name__ == "__main__":
    asyncio.run(process_bmw_documents())