#!/usr/bin/env python3
"""
Generate comprehensive HTML reports from already processed BMW documents
Uses cached results if available
"""

import asyncio
import json
import logging
import base64
from pathlib import Path
import sys
from datetime import datetime
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf.pdf_processor import PDFProcessor
from core.parsers.interfaces.data_models import SegmentType, TextSubtype, Document
from core.parsers.utils.segment_context_enhancer import SegmentContextEnhancer

# Import the report generation functions
from generate_comprehensive_bmw_report import (
    format_table_content, format_list_content, format_context, generate_document_html
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def load_or_process_document(pdf_path: Path, processor: PDFProcessor, cache_dir: Path):
    """Load document from cache or process it"""
    cache_file = cache_dir / f"{pdf_path.stem}_processed.pkl"
    
    if cache_file.exists():
        logger.info(f"📂 Loading cached document from {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # Process document
    logger.info(f"🔄 Processing document: {pdf_path}")
    document = await processor.parse_async(pdf_path)
    
    # Save to cache
    cache_dir.mkdir(exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(document, f)
        logger.info(f"💾 Saved to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
    
    return document


async def generate_reports_from_cache():
    """Generate comprehensive HTML reports using cached data where possible"""
    
    pdf_files = [
        "Preview_BMW_X5_G05.pdf",
        "Preview_BMW_8er_G14_G15.pdf",
        "Preview_BMW_1er_Sedan_CN.pdf",
        "Preview_BMW_3er_G20.pdf"
    ]
    
    # Configure processor (for cases where we need to process)
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
            "max_pages": 20
        },
        "image_analyzer_config": {
            "max_images": 10
        }
    }
    
    processor = PDFProcessor(config)
    cache_dir = Path("cache/processed_documents")
    
    for pdf_name in pdf_files:
        pdf_path = Path("data/input") / pdf_name
        
        if not pdf_path.exists():
            logger.warning(f"⚠️ PDF not found: {pdf_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Processing: {pdf_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Load or process document
            document = await load_or_process_document(pdf_path, processor, cache_dir)
            
            # Generate comprehensive HTML report
            output_path = Path(f"bmw_comprehensive_{pdf_name.replace('.pdf', '.html')}")
            generate_document_html(document, pdf_name, output_path)
            
            logger.info(f"✅ Successfully generated report for {pdf_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate index page
    generate_index_page(pdf_files)


def generate_index_page(pdf_files):
    """Generate an index page linking to all reports"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMW Documents - Comprehensive Analysis Index</title>
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
            padding: 30px;
            text-align: center;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .report-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }}
        .report-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 25px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .report-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        .report-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 15px;
        }}
        .report-link {{
            display: inline-block;
            background: #2196F3;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 15px;
            transition: background 0.2s;
        }}
        .report-link:hover {{
            background: #1976d2;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .status.available {{
            background: #4CAF50;
            color: white;
        }}
        .status.missing {{
            background: #f44336;
            color: white;
        }}
        .description {{
            color: #666;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 BMW Documents Analysis</h1>
        <p>Comprehensive Document Analysis with Context Enhancement</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="container">
        <h2>📊 Available Reports</h2>
        <div class="report-grid">
    """
    
    # Add card for each report
    for pdf_name in pdf_files:
        html_file = f"bmw_comprehensive_{pdf_name.replace('.pdf', '.html')}"
        html_path = Path(html_file)
        
        # Model descriptions
        descriptions = {
            "Preview_BMW_X5_G05.pdf": "BMW X5 (G05) - Luxury SUV with xDrive technology",
            "Preview_BMW_8er_G14_G15.pdf": "BMW 8 Series (G14/G15) - Premium Gran Coupé",
            "Preview_BMW_1er_Sedan_CN.pdf": "BMW 1 Series Sedan - Compact luxury for China market",
            "Preview_BMW_3er_G20.pdf": "BMW 3 Series (G20) - The ultimate sports sedan"
        }
        
        html_content += f"""
            <div class="report-card">
                <div class="report-title">{pdf_name}</div>
                <div class="description">{descriptions.get(pdf_name, 'BMW vehicle documentation')}</div>
        """
        
        if html_path.exists():
            html_content += f"""
                <span class="status available">✅ Report Available</span>
                <br>
                <a href="{html_file}" class="report-link">View Comprehensive Report →</a>
            """
        else:
            html_content += """
                <span class="status missing">⏳ Report Pending</span>
            """
        
        html_content += """
            </div>
        """
    
    html_content += """
        </div>
        
        <div style="margin-top: 50px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
            <h3>📋 Report Features</h3>
            <ul>
                <li>Complete document segmentation with page-by-page breakdown</li>
                <li>Context enhancement for tables, lists, and visual elements</li>
                <li>Embedded images and visual analysis results</li>
                <li>Formatted tables and lists for better readability</li>
                <li>Color-coded segments by type (tables, lists, visuals)</li>
                <li>Comprehensive statistics and metadata</li>
            </ul>
        </div>
    </div>
</body>
</html>
    """
    
    index_path = Path("bmw_reports_index.html")
    index_path.write_text(html_content)
    logger.info(f"📄 Index page saved to: {index_path}")


if __name__ == "__main__":
    asyncio.run(generate_reports_from_cache())