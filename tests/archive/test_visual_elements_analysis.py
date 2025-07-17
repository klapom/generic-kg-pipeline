#!/usr/bin/env python3
"""
Test to extract, save and analyze visual elements with VLM descriptions
"""

import asyncio
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image, ImageDraw
from io import BytesIO
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parsers.implementations.pdf import HybridPDFParser
from core.parsers.interfaces import SegmentType, VisualElementType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_visual_element_from_pdf(pdf_path: Path, page_num: int, bbox: List[int], output_path: Path):
    """Extract actual visual element from PDF using bbox coordinates"""
    try:
        import fitz  # PyMuPDF
        
        # Open PDF
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # Convert to 0-indexed
        
        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Convert bbox from 0-500 scale to page coordinates
        scale_x = page_width / 500.0
        scale_y = page_height / 500.0
        
        x1, y1, x2, y2 = bbox
        
        # Scale to page coordinates
        rect = fitz.Rect(
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        )
        
        # Render the region as image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=rect)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        # Add metadata overlay if image is large enough
        if img.width > 100 and img.height > 50:
            # Create overlay with metadata
            overlay_height = 30
            new_img = Image.new('RGB', (img.width, img.height + overlay_height), color='white')
            new_img.paste(img, (0, overlay_height))
            
            draw = ImageDraw.Draw(new_img)
            draw.rectangle([0, 0, img.width, overlay_height], fill='#e3f2fd')
            draw.text((5, 5), f"Page {page_num} - Bbox: {bbox}", fill='black')
            
            img = new_img
        
        img.save(output_path)
        doc.close()
        return True
        
    except ImportError:
        logger.warning("PyMuPDF not available, creating metadata preview instead")
        return create_metadata_preview(bbox, page_num, output_path)
    except Exception as e:
        logger.error(f"Failed to extract visual element from PDF: {e}")
        return create_metadata_preview(bbox, page_num, output_path)


def create_metadata_preview(bbox, page_num: int, output_path: Path):
    """Create a metadata preview when PDF extraction fails"""
    try:
        img_width = 400
        img_height = 300
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle([0, 0, img_width-1, img_height-1], outline='black', width=2)
        
        # Title area
        draw.rectangle([0, 0, img_width, 40], fill='#e3f2fd')
        draw.text((10, 10), f"Visual Element - Page {page_num}", fill='black')
        
        # Information
        y_pos = 60
        line_height = 25
        
        info_lines = [
            f"Page: {page_num}",
            f"Bbox (0-500): {bbox}",
            "⚠️ Could not extract image",
            "Using metadata preview"
        ]
        
        for line in info_lines:
            draw.text((20, y_pos), line, fill='black')
            y_pos += line_height
        
        # Draw visual representation of bbox
        mini_x = 50
        mini_y = 150
        mini_width = 300
        mini_height = 100
        
        draw.rectangle([mini_x, mini_y, mini_x + mini_width, mini_y + mini_height], outline='lightgray')
        draw.text((mini_x, mini_y - 20), "Relative position on page:", fill='gray')
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Scale bbox to mini view
            x = mini_x + (x1 * mini_width / 500)
            y = mini_y + (y1 * mini_height / 500)
            w = (x2 - x1) * mini_width / 500
            h = (y2 - y1) * mini_height / 500
            
            draw.rectangle([x, y, x + w, y + h], outline='red', width=2, fill='red')
        
        img.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Failed to create metadata preview: {e}")
        return False


def create_visual_element_image(ve, visual_segment, page_num: int, pdf_path: Path, output_path: Path):
    """Create an image for a visual element - either extracted from PDF or metadata preview"""
    
    # Try to extract actual visual element from PDF using bbox
    bbox = None
    if hasattr(ve, 'analysis_metadata') and ve.analysis_metadata and 'raw_bbox' in ve.analysis_metadata:
        bbox = ve.analysis_metadata['raw_bbox']
    elif hasattr(ve, 'bounding_box') and ve.bounding_box:
        # Convert bounding_box format to list
        bb = ve.bounding_box
        bbox = [bb.get('x', 0), bb.get('y', 0), bb.get('x', 0) + bb.get('width', 0), bb.get('y', 0) + bb.get('height', 0)]
    
    if bbox and len(bbox) == 4:
        # Try to extract actual image from PDF
        if extract_visual_element_from_pdf(pdf_path, page_num, bbox, output_path):
            return True
    
    # Fallback to metadata preview
    return create_metadata_preview(bbox, page_num, output_path)


async def analyze_visual_elements():
    """Analyze visual elements with and without VLM"""
    logger.info("=" * 80)
    logger.info("Visual Elements Analysis with VLM")
    logger.info("=" * 80)
    
    test_file = Path("data/input/Preview_BMW_3er_G20.pdf")
    output_dir = Path("tests/debugging/visual_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_dir / "extracted_images"
    images_dir.mkdir(exist_ok=True)
    
    # First pass: Parse without VLM
    logger.info("\n📖 Pass 1: Parsing without VLM to identify visual elements...")
    parser_no_vlm = HybridPDFParser(
        config={"pdfplumber_mode": 0},
        enable_vlm=False
    )
    
    doc_no_vlm = await parser_no_vlm.parse(test_file)
    
    logger.info(f"✅ Found {len(doc_no_vlm.visual_elements)} visual elements")
    
    # Extract visual elements as images
    logger.info("\n🖼️ Extracting visual elements as images...")
    visual_data = []
    
    for i, ve in enumerate(doc_no_vlm.visual_elements[:20]):  # First 20 for testing
        logger.info(f"\nProcessing visual element {i+1}/{min(20, len(doc_no_vlm.visual_elements))}")
        
        # Find corresponding visual segment
        visual_segment = None
        for seg in doc_no_vlm.segments:
            if seg.segment_type == SegmentType.VISUAL and ve.content_hash in seg.visual_references:
                visual_segment = seg
                break
        
        # Extract bounding box from visual element or segment
        bbox = None
        if hasattr(ve, 'bounding_box') and ve.bounding_box:
            bbox = (ve.bounding_box['x'], ve.bounding_box['y'], 
                   ve.bounding_box['x'] + ve.bounding_box['width'],
                   ve.bounding_box['y'] + ve.bounding_box['height'])
        elif hasattr(ve, 'analysis_metadata') and ve.analysis_metadata and 'bbox' in ve.analysis_metadata:
            bbox = ve.analysis_metadata['bbox']
        
        # Try to extract from SmolDocling raw output
        if not bbox and visual_segment:
            # Look for location info in metadata
            logger.debug(f"Visual element metadata: {ve.analysis_metadata}")
        
        image_path = images_dir / f"visual_{i+1}_page{ve.page_or_slide}_{ve.element_type.value}.png"
        
        # Create visual element image (extracted from PDF or metadata preview)
        create_visual_element_image(ve, visual_segment, ve.page_or_slide, test_file, image_path)
        
        visual_info = {
            "index": i + 1,
            "element_type": ve.element_type.value,
            "page": ve.page_or_slide,
            "content_hash": ve.content_hash,
            "image_file": str(image_path.name),
            "has_bbox": bbox is not None,
            "bbox": bbox,
            "segment_content": visual_segment.content if visual_segment else None,
            "vlm_description": None,  # Will be filled in second pass
            "extracted_from_pdf": bbox is not None and len(bbox) == 4
        }
        
        logger.info(f"✅ Element {i+1}: {ve.element_type.value} on page {ve.page_or_slide}, bbox: {bbox is not None}")
        
        visual_data.append(visual_info)
    
    # Second pass: Parse with VLM (if available)
    logger.info("\n📖 Pass 2: Parsing with VLM to get descriptions...")
    
    vlm_analysis_results = {}
    
    try:
        parser_vlm = HybridPDFParser(
            config={"pdfplumber_mode": 0},
            enable_vlm=True
        )
        
        logger.info("🤖 Starting VLM analysis...")
        doc_vlm = await parser_vlm.parse(test_file)
        
        logger.info(f"🔍 VLM found {len(doc_vlm.visual_elements)} visual elements with descriptions")
        
        # Match VLM descriptions to visual elements
        for i, ve_vlm in enumerate(doc_vlm.visual_elements[:20]):
            if i < len(visual_data):
                vlm_desc = ve_vlm.vlm_description
                confidence = getattr(ve_vlm, 'confidence', 0.0)
                
                visual_data[i]["vlm_description"] = vlm_desc
                visual_data[i]["vlm_confidence"] = confidence
                
                # Log detailed VLM analysis results
                logger.info(f"\n🎯 VLM Analysis Element {i+1}:")
                logger.info(f"   - Page: {ve_vlm.page_or_slide}")
                logger.info(f"   - Type: {ve_vlm.element_type.value}")
                logger.info(f"   - Hash: {ve_vlm.content_hash[:16]}...")
                logger.info(f"   - Confidence: {confidence:.2f}")
                logger.info(f"   - Description: {vlm_desc[:100]}{'...' if vlm_desc and len(vlm_desc) > 100 else ''}")
                
                vlm_analysis_results[ve_vlm.content_hash] = {
                    "description": vlm_desc,
                    "confidence": confidence,
                    "element_type": ve_vlm.element_type.value,
                    "page": ve_vlm.page_or_slide
                }
                
                # Also check if segment was updated
                for seg in doc_vlm.segments:
                    if seg.segment_type == SegmentType.VISUAL and ve_vlm.content_hash in seg.visual_references:
                        if not seg.content.startswith("["):  # Not a placeholder
                            visual_data[i]["vlm_segment_content"] = seg.content
                            logger.info(f"   - Updated segment: {seg.content[:100]}{'...' if len(seg.content) > 100 else ''}")
                        break
        
        logger.info(f"\n✅ VLM analysis completed! {len(vlm_analysis_results)} elements analyzed")
        
        # Log VLM performance summary
        if vlm_analysis_results:
            confidences = [r["confidence"] for r in vlm_analysis_results.values()]
            avg_confidence = sum(confidences) / len(confidences)
            logger.info(f"📊 VLM Analysis Summary:")
            logger.info(f"   - Average confidence: {avg_confidence:.2f}")
            logger.info(f"   - Min confidence: {min(confidences):.2f}")
            logger.info(f"   - Max confidence: {max(confidences):.2f}")
        
    except Exception as e:
        logger.error(f"❌ VLM pass failed: {e}")
        logger.exception("VLM error details:")
    
    # Create comprehensive report
    report = {
        "test_file": str(test_file),
        "timestamp": datetime.now().isoformat(),
        "total_visual_elements": len(doc_no_vlm.visual_elements),
        "analyzed_count": len(visual_data),
        "vlm_analysis_results": vlm_analysis_results,
        "visual_elements": visual_data
    }
    
    # Save report
    report_file = output_dir / f"visual_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Create HTML report for easy viewing
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visual Elements Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .visual-item {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; }}
            .visual-item img {{ max-width: 400px; border: 1px solid #ddd; }}
            .info {{ margin-left: 420px; }}
            .placeholder {{ background: #f0f0f0; padding: 5px; margin: 5px 0; }}
            .vlm-desc {{ background: #e8f5e9; padding: 10px; margin: 10px 0; border-left: 4px solid #4caf50; }}
            .confidence {{ color: #4caf50; font-weight: bold; }}
            .low-confidence {{ color: #ff9800; }}
            .high-confidence {{ color: #4caf50; }}
            .bbox-info {{ background: #f3e5f5; padding: 5px; margin: 5px 0; font-family: monospace; }}
            .extracted-badge {{ background: #4caf50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
            .metadata-badge {{ background: #ff9800; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>Visual Elements Analysis Report</h1>
        <p>File: {test_file.name}</p>
        <p>Total Visual Elements: {len(doc_no_vlm.visual_elements)}</p>
        <p>Analyzed: {len(visual_data)}</p>
        <hr>
    """
    
    for item in visual_data:
        # Determine image source type badge
        source_badge = ""
        if item.get('extracted_from_pdf'):
            source_badge = '<span class="extracted-badge">📷 Extracted from PDF</span>'
        else:
            source_badge = '<span class="metadata-badge">📊 Metadata Preview</span>'
        
        # Determine confidence styling
        vlm_confidence = item.get('vlm_confidence', 0.0)
        confidence_class = "high-confidence" if vlm_confidence > 0.7 else "low-confidence" if vlm_confidence > 0.3 else "confidence"
        
        html_content += f"""
        <div class="visual-item">
            <img src="extracted_images/{item['image_file']}" style="float: left; margin-right: 20px;">
            <div class="info">
                <h3>Element {item['index']} {source_badge}</h3>
                <p><strong>Type:</strong> {item['element_type']}</p>
                <p><strong>Page:</strong> {item['page']}</p>
                <p><strong>Hash:</strong> {item['content_hash'][:16]}...</p>
                
                {"<div class='bbox-info'><strong>BBox:</strong> " + str(item['bbox']) + "</div>" if item['bbox'] else '<p><strong>BBox:</strong> Not available</p>'}
                
                <div class="placeholder">
                    <strong>📝 Original Placeholder:</strong><br>
                    {item['segment_content'] or 'N/A'}
                </div>
        """
        
        if item.get('vlm_description'):
            html_content += f"""
                <div class="vlm-desc">
                    <strong>🤖 VLM Analysis:</strong>
                    <span class="{confidence_class}">Confidence: {vlm_confidence:.2f}</span><br><br>
                    {item['vlm_description']}
                </div>
            """
        elif item.get('vlm_segment_content'):
            html_content += f"""
                <div class="vlm-desc">
                    <strong>🤖 VLM Updated Segment:</strong><br>
                    {item['vlm_segment_content']}
                </div>
            """
        elif vlm_analysis_results:
            html_content += f"""
                <div class="vlm-desc" style="background: #fff3e0; border-left-color: #ff9800;">
                    <strong>⚠️ VLM Analysis:</strong> No description available
                </div>
            """
        
        html_content += """
            </div>
            <div style="clear: both;"></div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    html_file = output_dir / "visual_analysis_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"\n📊 Analysis complete! Results saved to:")
    logger.info(f"   - JSON Report: {report_file}")
    logger.info(f"   - HTML Report: {html_file}")
    logger.info(f"   - Images: {images_dir}/")
    
    # Print summary
    logger.info("\n📈 Summary:")
    logger.info(f"   - Total visual elements: {len(doc_no_vlm.visual_elements)}")
    logger.info(f"   - Unique types: {set(ve.element_type.value for ve in doc_no_vlm.visual_elements)}")
    
    # Count by type
    type_counts = {}
    for ve in doc_no_vlm.visual_elements:
        type_counts[ve.element_type.value] = type_counts.get(ve.element_type.value, 0) + 1
    
    logger.info("\n   Distribution by type:")
    for element_type, count in sorted(type_counts.items()):
        logger.info(f"     - {element_type}: {count}")


if __name__ == "__main__":
    asyncio.run(analyze_visual_elements())