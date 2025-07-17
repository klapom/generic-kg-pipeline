"""
Enhanced Pipeline Debugger with Full Content Display

Provides comprehensive debugging with actual content, images, and VLM comparisons.
Based on successful test cases from VLM comparison tests.
"""

import json
import base64
import logging
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
import fitz  # PyMuPDF

from core.parsers import Document, Segment, VisualElement
from core.content_chunker import ContextualChunk, ChunkingResult
from core.pipeline_debugger import (
    DebugLevel, PipelineDebugConfig, SegmentDebugInfo, 
    ChunkDebugInfo, PipelineDebugData
)


logger = logging.getLogger(__name__)


@dataclass
class EnhancedSegmentDebugInfo(SegmentDebugInfo):
    """Enhanced segment info with full content and images"""
    full_content: str = ""
    image_data: Optional[str] = None  # base64 encoded
    vlm_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # model -> result
    smoldocling_raw: Optional[Dict[str, Any]] = None  # Raw SmolDocling output


@dataclass 
class EnhancedPipelineDebugData(PipelineDebugData):
    """Enhanced debug data with full content"""
    enhanced_segments: List[EnhancedSegmentDebugInfo] = field(default_factory=list)
    page_images: Dict[int, str] = field(default_factory=dict)  # page_num -> base64 image
    smoldocling_pages: List[Dict[str, Any]] = field(default_factory=list)  # Raw pages from SmolDocling


class EnhancedPipelineDebugger:
    """Enhanced debugger that captures full content and creates detailed HTML reports"""
    
    def __init__(self, config: PipelineDebugConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_debug_data: Optional[EnhancedPipelineDebugData] = None
        self.pdf_path: Optional[Path] = None
        
        # Ensure output directory exists
        if config.debug_level != DebugLevel.NONE:
            config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_document_processing(self, document_path: Path, document_id: str) -> None:
        """Start tracking a new document processing"""
        if self.config.debug_level == DebugLevel.NONE:
            return
            
        self.pdf_path = document_path
        self.current_debug_data = EnhancedPipelineDebugData(
            document_id=document_id,
            document_path=document_path,
            start_time=datetime.now()
        )
        
        # Extract page images if PDF
        if document_path.suffix.lower() == '.pdf' and self.config.include_images:
            self._extract_page_images(document_path)
        
        self.logger.info(f"Started enhanced debugging for: {document_path.name}")
    
    def _extract_page_images(self, pdf_path: Path) -> None:
        """Extract all pages as images from PDF"""
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to base64
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode()
                self.current_debug_data.page_images[page_num + 1] = img_base64
                
            doc.close()
            self.logger.info(f"Extracted {len(self.current_debug_data.page_images)} page images")
        except Exception as e:
            self.logger.error(f"Failed to extract page images: {e}")
    
    def track_smoldocling_output(self, pages: List[Dict[str, Any]]) -> None:
        """Track raw SmolDocling output"""
        if not self.current_debug_data:
            return
            
        self.current_debug_data.smoldocling_pages = pages
        self.logger.info(f"Tracked {len(pages)} SmolDocling pages")
    
    def track_parsing_complete(self, document: Document, parsing_time: float, 
                              smoldocling_result: Optional[Dict[str, Any]] = None) -> None:
        """Track completion of parsing stage with full content"""
        if not self.current_debug_data:
            return
            
        self.current_debug_data.parsing_time = parsing_time
        
        # Track basic statistics
        self.current_debug_data.stats.update({
            "total_segments": len(document.segments),
            "total_visual_elements": len(document.visual_elements),
            "page_count": document.metadata.page_count,
            "document_type": document.metadata.document_type.value
        })
        
        # Track enhanced segments with full content
        if self.config.track_segments:
            for segment in document.segments:
                enhanced_info = EnhancedSegmentDebugInfo(
                    segment=segment,
                    full_content=segment.content,  # Store full content
                    metadata={
                        "page": segment.page_number,
                        "type": segment.segment_type,
                        "has_visual": bool(segment.visual_references) if hasattr(segment, 'visual_references') else False,
                        "char_count": len(segment.content)
                    }
                )
                
                # Link to SmolDocling raw data if available
                if smoldocling_result and hasattr(segment, 'page_number'):
                    page_idx = segment.page_number - 1
                    if 0 <= page_idx < len(smoldocling_result.get('pages', [])):
                        enhanced_info.smoldocling_raw = smoldocling_result['pages'][page_idx]
                
                self.current_debug_data.enhanced_segments.append(enhanced_info)
        
        # Track visual elements with image extraction
        for visual in document.visual_elements:
            self._extract_visual_image(visual)
    
    def _extract_visual_image(self, visual: VisualElement) -> None:
        """Extract image for a visual element"""
        if not self.pdf_path or not hasattr(visual, 'bounding_box'):
            return
            
        try:
            # Extract image from PDF using bounding box
            if hasattr(visual, 'page_or_slide') and visual.bounding_box:
                doc = fitz.open(str(self.pdf_path))
                page = doc[visual.page_or_slide - 1]
                
                # Get page dimensions for scaling
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                
                # Scale from SmolDocling's 0-500 coordinate system to page coordinates
                scale_x = page_width / 500.0
                scale_y = page_height / 500.0
                
                # Convert bbox to fitz.Rect with proper scaling
                bbox = visual.bounding_box
                if isinstance(bbox, dict):
                    x0 = bbox.get('x0', 0) * scale_x
                    y0 = bbox.get('y0', 0) * scale_y
                    x1 = bbox.get('x1', 100) * scale_x
                    y1 = bbox.get('y1', 100) * scale_y
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x0 = bbox[0] * scale_x
                    y0 = bbox[1] * scale_y
                    x1 = bbox[2] * scale_x
                    y1 = bbox[3] * scale_y
                else:
                    self.logger.warning(f"Invalid bbox format: {bbox}")
                    x0, y0, x1, y1 = 0, 0, page_width, page_height
                
                rect = fitz.Rect(x0, y0, x1, y1)
                
                # Extract with zoom
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=rect)
                
                # Convert to base64
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode()
                
                # Store in visual elements tracking
                self.current_debug_data.visual_elements.append({
                    "id": visual.content_hash,
                    "type": visual.element_type.value if hasattr(visual.element_type, 'value') else str(visual.element_type),
                    "page": visual.page_or_slide,
                    "bbox": visual.bounding_box,
                    "has_image": True,
                    "image_base64": img_base64
                })
                
                doc.close()
        except Exception as e:
            self.logger.error(f"Failed to extract visual image: {e}")
    
    def track_vlm_processing_multi(self, visual_id: str, results: Dict[str, Dict[str, Any]]) -> None:
        """Track VLM processing results from multiple models"""
        if not self.current_debug_data:
            return
            
        # Find the enhanced segment for this visual
        for seg_info in self.current_debug_data.enhanced_segments:
            if hasattr(seg_info.segment, 'visual_references') and visual_id in seg_info.segment.visual_references:
                seg_info.vlm_results = results
                seg_info.vlm_applied = True
                
                # Calculate average confidence
                confidences = [r.get('confidence', 0) for r in results.values()]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                seg_info.vlm_confidence = avg_confidence
                
                # Track best model
                best_model = max(results.items(), key=lambda x: x[1].get('confidence', 0))
                seg_info.vlm_model = best_model[0]
                seg_info.vlm_description = best_model[1].get('description', '')
                
                break
        
        # Update VLM processing time
        total_time = sum(r.get('processing_time', 0) for r in results.values())
        self.current_debug_data.vlm_processing_time += total_time
    
    def generate_enhanced_html_report(self) -> Path:
        """Generate comprehensive HTML report with full content display"""
        if not self.current_debug_data:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_debug_data.document_path.stem}_enhanced_analysis_{timestamp}.html"
        output_file = self.config.output_dir / filename
        
        html_content = self._create_enhanced_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def _create_enhanced_html_report(self) -> str:
        """Create the enhanced HTML report content"""
        doc_name = self.current_debug_data.document_path.name
        
        html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Pipeline Analysis - {doc_name}</title>
    <style>
        {self._get_enhanced_css_styles()}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Enhanced Pipeline Analysis Report</h1>
        <p>{doc_name} - {self.current_debug_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <!-- Summary Section -->
        {self._generate_summary_section()}
        
        <!-- Page-by-Page Analysis -->
        {self._generate_page_analysis()}
        
        <!-- Chunks Analysis -->
        {self._generate_chunks_section()}
    </div>
    
    <script>
        {self._get_enhanced_javascript()}
    </script>
</body>
</html>"""
        return html
    
    def _generate_page_analysis(self) -> str:
        """Generate page-by-page analysis with full content"""
        html_parts = []
        
        # Group segments by page
        segments_by_page = {}
        for seg_info in self.current_debug_data.enhanced_segments:
            page = seg_info.segment.page_number or 0
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(seg_info)
        
        # Generate analysis for each page
        for page_num in sorted(segments_by_page.keys()):
            if page_num == 0:
                continue  # Skip segments without page number
                
            page_segments = segments_by_page[page_num]
            
            html_parts.append(f"""
            <div class="section page-section">
                <h2>📄 Page {page_num} Analysis</h2>
                
                <!-- Page Image -->
                {self._generate_page_image_section(page_num)}
                
                <!-- SmolDocling Content -->
                <div class="smoldocling-content">
                    <h3>📝 Extracted Content (SmolDocling)</h3>
                    {self._generate_smoldocling_content(page_segments)}
                </div>
                
                <!-- Visual Elements with VLM -->
                {self._generate_visual_elements_section(page_num, page_segments)}
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_page_image_section(self, page_num: int) -> str:
        """Generate page image section"""
        if page_num in self.current_debug_data.page_images:
            return f"""
            <div class="page-image-container">
                <h3>📸 Original Page</h3>
                <img src="data:image/png;base64,{self.current_debug_data.page_images[page_num]}" 
                     class="page-image" alt="Page {page_num}">
            </div>
            """
        return ""
    
    def _generate_smoldocling_content(self, segments: List[EnhancedSegmentDebugInfo]) -> str:
        """Generate SmolDocling extracted content"""
        html_parts = []
        
        for seg_info in segments:
            segment_class = "text-segment"
            if seg_info.segment.segment_type == "heading":
                segment_class = "heading-segment"
            elif seg_info.segment.segment_type == "table":
                segment_class = "table-segment"
                
            html_parts.append(f"""
            <div class="segment {segment_class}">
                <div class="segment-header">
                    <span class="segment-type">{seg_info.segment.segment_type}</span>
                    <span class="segment-chars">{len(seg_info.full_content)} chars</span>
                </div>
                <div class="segment-content">
                    {self._escape_html(seg_info.full_content)}
                </div>
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_visual_elements_section(self, page_num: int, segments: List[EnhancedSegmentDebugInfo]) -> str:
        """Generate visual elements with VLM comparisons"""
        visual_segments = [s for s in segments if s.vlm_applied]
        
        if not visual_segments:
            return ""
            
        html_parts = ["""
        <div class="visual-elements-section">
            <h3>🖼️ Visual Elements with VLM Analysis</h3>
        """]
        
        for seg_info in visual_segments:
            # Find visual element data
            visual_data = None
            for v in self.current_debug_data.visual_elements:
                if v.get('page') == page_num:
                    visual_data = v
                    break
            
            html_parts.append(f"""
            <div class="visual-element-container">
                <!-- Visual Image -->
                {self._generate_visual_image(visual_data)}
                
                <!-- VLM Comparisons -->
                <div class="vlm-comparison-grid">
                    {self._generate_vlm_comparisons(seg_info.vlm_results)}
                </div>
            </div>
            """)
        
        html_parts.append("</div>")
        return '\n'.join(html_parts)
    
    def _generate_visual_image(self, visual_data: Dict[str, Any]) -> str:
        """Generate visual element image"""
        if visual_data and visual_data.get('image_base64'):
            return f"""
            <div class="visual-image">
                <img src="data:image/png;base64,{visual_data['image_base64']}" 
                     alt="Visual Element" class="visual-img">
            </div>
            """
        return ""
    
    def _generate_vlm_comparisons(self, vlm_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate VLM comparison grid like in successful tests"""
        if not vlm_results:
            return ""
            
        html_parts = []
        
        # Order: Qwen, LLaVA, Pixtral
        model_order = ['qwen2.5-vl', 'llava', 'pixtral']
        
        for model in model_order:
            if model in vlm_results:
                result = vlm_results[model]
                status_class = "success" if result.get('success', True) else "failure"
                
                html_parts.append(f"""
                <div class="vlm-card {status_class}">
                    <div class="vlm-header">
                        <h4>{self._get_model_display_name(model)}</h4>
                        <span class="confidence">Confidence: {result.get('confidence', 0):.2%}</span>
                    </div>
                    <div class="vlm-content">
                        <div class="description">
                            <strong>Description:</strong><br>
                            {self._escape_html(result.get('description', 'No description'))}
                        </div>
                        {self._generate_ocr_section(result.get('ocr_text', ''))}
                        {self._generate_extracted_data_section(result.get('extracted_data', {}))}
                    </div>
                    <div class="vlm-footer">
                        <span class="processing-time">⏱️ {result.get('processing_time', 0):.2f}s</span>
                    </div>
                </div>
                """)
        
        return '\n'.join(html_parts)
    
    def _get_model_display_name(self, model_key: str) -> str:
        """Get display name for model"""
        names = {
            'qwen2.5-vl': '🎯 Qwen2.5-VL-7B',
            'llava': '👁️ LLaVA-1.6-Mistral-7B',
            'pixtral': '🔍 Pixtral-12B'
        }
        return names.get(model_key, model_key)
    
    def _generate_ocr_section(self, ocr_text: str) -> str:
        """Generate OCR text section"""
        if not ocr_text:
            return ""
        return f"""
        <div class="ocr-text">
            <strong>OCR Text:</strong><br>
            <code>{self._escape_html(ocr_text)}</code>
        </div>
        """
    
    def _generate_extracted_data_section(self, data: Dict[str, Any]) -> str:
        """Generate extracted data section"""
        if not data:
            return ""
        return f"""
        <div class="extracted-data">
            <strong>Extracted Data:</strong><br>
            <pre>{json.dumps(data, indent=2)}</pre>
        </div>
        """
    
    def _generate_chunks_section(self) -> str:
        """Generate chunks analysis section"""
        if not hasattr(self.current_debug_data, 'chunks') or not self.current_debug_data.chunks:
            return ""
            
        return f"""
        <div class="section">
            <h2>📦 Contextual Chunks Analysis</h2>
            <div class="chunks-grid">
                {self._generate_chunks_html()}
            </div>
        </div>
        """
    
    def _generate_chunks_html(self) -> str:
        """Generate HTML for chunks"""
        html_parts = []
        
        for chunk_info in self.current_debug_data.chunks[:20]:  # Limit to first 20
            html_parts.append(f"""
            <div class="chunk-card">
                <h4>Chunk {chunk_info.chunk.chunk_id}</h4>
                <div class="chunk-meta">
                    <span>📦 Type: {chunk_info.chunk.chunk_type.value}</span>
                    <span>🔢 Tokens: {chunk_info.chunk.token_count}</span>
                    <span>📄 Sources: {len(chunk_info.source_segments)} segments</span>
                </div>
                <div class="chunk-content">
                    {self._escape_html(chunk_info.chunk.content[:500])}{'...' if len(chunk_info.chunk.content) > 500 else ''}
                </div>
                {self._generate_context_inheritance(chunk_info)}
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_context_inheritance(self, chunk_info: ChunkDebugInfo) -> str:
        """Generate context inheritance display"""
        if not chunk_info.context_inheritance_applied:
            return ""
        return f"""
        <div class="context-inheritance">
            <strong>🔗 Inherited Context:</strong>
            <div class="inherited-content">
                {self._escape_html(chunk_info.chunk.inherited_context[:200])}...
            </div>
        </div>
        """
    
    def _generate_summary_section(self) -> str:
        """Generate summary section"""
        vlm_segments = sum(1 for s in self.current_debug_data.enhanced_segments if s.vlm_applied)
        
        return f"""
        <div class="section">
            <h2>📊 Processing Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>⏱️ Performance</h3>
                    <p><strong>Total Time:</strong> {self.current_debug_data.total_time:.2f}s</p>
                    <p><strong>Parsing:</strong> {self.current_debug_data.parsing_time:.2f}s</p>
                    <p><strong>VLM Processing:</strong> {self.current_debug_data.vlm_processing_time:.2f}s</p>
                    <p><strong>Chunking:</strong> {self.current_debug_data.chunking_time:.2f}s</p>
                </div>
                
                <div class="stat-card">
                    <h3>📄 Document Stats</h3>
                    <p><strong>Pages:</strong> {self.current_debug_data.stats.get('page_count', 0)}</p>
                    <p><strong>Segments:</strong> {self.current_debug_data.stats.get('total_segments', 0)}</p>
                    <p><strong>Visual Elements:</strong> {self.current_debug_data.stats.get('total_visual_elements', 0)}</p>
                    <p><strong>Chunks:</strong> {self.current_debug_data.stats.get('total_chunks', 0)}</p>
                </div>
                
                <div class="stat-card">
                    <h3>🤖 VLM Analysis</h3>
                    <p><strong>Processed Elements:</strong> {vlm_segments}</p>
                    <p><strong>Models Used:</strong> Qwen2.5-VL, LLaVA, Pixtral</p>
                </div>
            </div>
        </div>
        """
    
    def _get_enhanced_css_styles(self) -> str:
        """Get enhanced CSS styles"""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .section {
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .page-section {
            border-left: 4px solid #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        /* Page Image */
        .page-image-container {
            margin-bottom: 2rem;
        }
        
        .page-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* SmolDocling Content */
        .smoldocling-content {
            margin-bottom: 2rem;
        }
        
        .segment {
            background: #f8f9fa;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #e9ecef;
        }
        
        .heading-segment {
            border-left-color: #667eea;
            font-weight: bold;
        }
        
        .table-segment {
            border-left-color: #28a745;
        }
        
        .segment-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .segment-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        /* Visual Elements */
        .visual-elements-section {
            margin-top: 2rem;
        }
        
        .visual-element-container {
            background: #f8f9fa;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        
        .visual-image {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .visual-img {
            max-width: 600px;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        /* VLM Comparison Grid */
        .vlm-comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1rem;
        }
        
        .vlm-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
        }
        
        .vlm-card.success {
            border-color: #28a745;
        }
        
        .vlm-card.failure {
            border-color: #dc3545;
        }
        
        .vlm-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e9ecef;
        }
        
        .vlm-header h4 {
            margin: 0;
            color: #667eea;
        }
        
        .confidence {
            font-size: 0.9rem;
            color: #28a745;
            font-weight: bold;
        }
        
        .vlm-content {
            margin-bottom: 1rem;
        }
        
        .description {
            margin-bottom: 1rem;
            line-height: 1.5;
        }
        
        .ocr-text {
            background: #f8f9fa;
            padding: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .ocr-text code {
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        .extracted-data {
            background: #f8f9fa;
            padding: 0.5rem;
            border-radius: 4px;
        }
        
        .extracted-data pre {
            margin: 0;
            font-size: 0.8rem;
            overflow-x: auto;
        }
        
        .vlm-footer {
            text-align: right;
            font-size: 0.9rem;
            color: #666;
        }
        
        /* Chunks */
        .chunks-grid {
            display: grid;
            gap: 1rem;
        }
        
        .chunk-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 3px solid #764ba2;
        }
        
        .chunk-meta {
            display: flex;
            gap: 1rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: #666;
        }
        
        .chunk-content {
            background: white;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
        }
        
        .context-inheritance {
            background: #e7f3ff;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 4px;
        }
        
        .inherited-content {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            font-style: italic;
        }
        """
    
    def _get_enhanced_javascript(self) -> str:
        """Get enhanced JavaScript"""
        return """
        // Toggle visibility of long content
        function toggleContent(elementId) {
            const element = document.getElementById(elementId);
            element.style.display = element.style.display === 'none' ? 'block' : 'none';
        }
        
        // Zoom image on click
        document.addEventListener('DOMContentLoaded', function() {
            const images = document.querySelectorAll('.page-image, .visual-img');
            images.forEach(img => {
                img.style.cursor = 'zoom-in';
                img.addEventListener('click', function() {
                    if (this.style.maxWidth === '100%') {
                        this.style.maxWidth = 'none';
                        this.style.cursor = 'zoom-out';
                    } else {
                        this.style.maxWidth = '100%';
                        this.style.cursor = 'zoom-in';
                    }
                });
            });
        });
        """
    
    def track_error(self, error_type: str, error: Exception) -> None:
        """Track errors during processing"""
        if not self.current_debug_data:
            return
        
        if not hasattr(self.current_debug_data, 'errors'):
            self.current_debug_data.errors = []
        
        self.current_debug_data.errors.append({
            'type': error_type,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.error(f"Tracked error in {error_type}: {error}")
    
    def end_document_processing(self) -> None:
        """End tracking for current document"""
        if self.current_debug_data:
            self.current_debug_data.end_time = datetime.now()
            self.current_debug_data.total_time = (
                self.current_debug_data.end_time - self.current_debug_data.start_time
            ).total_seconds()
    
    def track_chunking_complete(self, chunking_result: ChunkingResult, chunking_time: float) -> None:
        """Track completion of chunking stage"""
        if not self.current_debug_data:
            return
        
        self.current_debug_data.chunking_time = chunking_time
        self.current_debug_data.stats['total_chunks'] = len(chunking_result.contextual_chunks)
        
        # Store chunks for report generation
        if not hasattr(self.current_debug_data, 'chunks'):
            self.current_debug_data.chunks = []
        
        for chunk in chunking_result.contextual_chunks[:20]:  # Limit to first 20 for display
            chunk_info = ChunkDebugInfo(
                chunk=chunk,
                source_segments=[],  # Would need to track this separately
                context_inheritance_applied=bool(chunk.inherited_context)
            )
            self.current_debug_data.chunks.append(chunk_info)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        if not text:
            return ""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))