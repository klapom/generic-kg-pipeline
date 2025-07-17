"""
Pipeline Debugger and Visualization Module

Provides comprehensive debugging and visualization capabilities for the production pipeline.
Tracks segments, chunks, VLM descriptions and generates HTML reports for analysis.
"""

import json
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.parsers import Document, Segment, VisualElement
from core.content_chunker import ContextualChunk, ChunkingResult


class DebugLevel(Enum):
    """Debug levels for pipeline execution"""
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"


@dataclass
class PipelineDebugConfig:
    """Configuration for pipeline debugging"""
    debug_level: DebugLevel = DebugLevel.NONE
    generate_html_report: bool = False
    track_segments: bool = True
    track_chunks: bool = True
    track_vlm_descriptions: bool = True
    save_intermediate_results: bool = False
    output_dir: Path = field(default_factory=lambda: Path("data/debug"))
    include_images: bool = True
    max_content_preview: int = 500  # Max characters to show in previews


@dataclass
class SegmentDebugInfo:
    """Debug information for a segment"""
    segment: Segment
    processing_time: float = 0.0
    vlm_applied: bool = False
    vlm_model: Optional[str] = None
    vlm_description: Optional[str] = None
    vlm_confidence: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkDebugInfo:
    """Debug information for a chunk"""
    chunk: ContextualChunk
    source_segments: List[str] = field(default_factory=list)  # segment IDs
    processing_time: float = 0.0
    context_inheritance_applied: bool = False
    inherited_from: Optional[str] = None  # chunk ID
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDebugData:
    """Complete debug data for a document processing run"""
    document_id: str
    document_path: Path
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Stage timings
    parsing_time: float = 0.0
    vlm_processing_time: float = 0.0
    chunking_time: float = 0.0
    total_time: float = 0.0
    
    # Data tracking
    segments: List[SegmentDebugInfo] = field(default_factory=list)
    chunks: List[ChunkDebugInfo] = field(default_factory=list)
    visual_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


class PipelineDebugger:
    """Main debugger for pipeline execution"""
    
    def __init__(self, config: PipelineDebugConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_debug_data: Optional[PipelineDebugData] = None
        
        # Ensure output directory exists
        if config.debug_level != DebugLevel.NONE:
            config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_document_processing(self, document_path: Path, document_id: str) -> None:
        """Start tracking a new document processing"""
        if self.config.debug_level == DebugLevel.NONE:
            return
            
        self.current_debug_data = PipelineDebugData(
            document_id=document_id,
            document_path=document_path,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Started debugging document: {document_path.name}")
    
    def track_parsing_complete(self, document: Document, parsing_time: float) -> None:
        """Track completion of parsing stage"""
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
        
        # Track segments if enabled
        if self.config.track_segments:
            for segment in document.segments:
                debug_info = SegmentDebugInfo(
                    segment=segment,
                    metadata={
                        "page": segment.page_number,
                        "type": segment.segment_type,
                        "has_visual": bool(segment.visual_references) if hasattr(segment, 'visual_references') else False
                    }
                )
                self.current_debug_data.segments.append(debug_info)
        
        # Track visual elements
        for visual in document.visual_elements:
            self.current_debug_data.visual_elements.append({
                "id": visual.content_hash,
                "type": visual.element_type.value if hasattr(visual.element_type, 'value') else str(visual.element_type),
                "page": visual.page_or_slide,
                "bbox": visual.bounding_box if hasattr(visual, 'bounding_box') else None,
                "has_image": visual.raw_data is not None
            })
    
    def track_vlm_processing(self, segment_id: str, model: str, description: str, 
                           confidence: float, processing_time: float) -> None:
        """Track VLM processing for a segment"""
        if not self.current_debug_data or not self.config.track_vlm_descriptions:
            return
            
        # Find the segment debug info
        for seg_info in self.current_debug_data.segments:
            if str(seg_info.segment.segment_index) == str(segment_id) or \
               (hasattr(seg_info.segment, 'visual_references') and segment_id in seg_info.segment.visual_references):
                seg_info.vlm_applied = True
                seg_info.vlm_model = model
                seg_info.vlm_description = description
                seg_info.vlm_confidence = confidence
                seg_info.processing_time += processing_time
                break
        
        self.current_debug_data.vlm_processing_time += processing_time
    
    def track_chunking_complete(self, chunking_result: ChunkingResult, chunking_time: float) -> None:
        """Track completion of chunking stage"""
        if not self.current_debug_data:
            return
            
        self.current_debug_data.chunking_time = chunking_time
        
        # Update statistics
        self.current_debug_data.stats.update({
            "total_chunks": len(chunking_result.contextual_chunks),
            "total_context_groups": len(chunking_result.context_groups),
            "avg_chunk_tokens": sum(c.token_count for c in chunking_result.contextual_chunks) / len(chunking_result.contextual_chunks) if chunking_result.contextual_chunks else 0
        })
        
        # Track chunks if enabled
        if self.config.track_chunks:
            for chunk in chunking_result.contextual_chunks:
                # Find source segments
                source_segment_ids = []
                for seg in chunk.source_segments:
                    if hasattr(seg, 'segment_index'):
                        source_segment_ids.append(str(seg.segment_index))
                
                debug_info = ChunkDebugInfo(
                    chunk=chunk,
                    source_segments=source_segment_ids,
                    context_inheritance_applied=bool(chunk.inherited_context),
                    metadata={
                        "chunk_type": chunk.chunk_type.value,
                        "token_count": chunk.token_count,
                        "generates_context": chunk.generates_context
                    }
                )
                self.current_debug_data.chunks.append(debug_info)
    
    def track_error(self, stage: str, error: Exception) -> None:
        """Track an error during processing"""
        if not self.current_debug_data:
            return
            
        self.current_debug_data.errors.append({
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def track_warning(self, stage: str, message: str) -> None:
        """Track a warning during processing"""
        if not self.current_debug_data:
            return
            
        self.current_debug_data.warnings.append({
            "stage": stage,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def end_document_processing(self) -> Optional[Path]:
        """End tracking and generate reports"""
        if not self.current_debug_data:
            return None
            
        self.current_debug_data.end_time = datetime.now()
        self.current_debug_data.total_time = (
            self.current_debug_data.end_time - self.current_debug_data.start_time
        ).total_seconds()
        
        # Save debug data
        debug_file = None
        if self.config.save_intermediate_results:
            debug_file = self._save_debug_data()
        
        # Generate HTML report
        html_file = None
        if self.config.generate_html_report:
            html_file = self._generate_html_report()
        
        self.logger.info(f"Completed debugging for document: {self.current_debug_data.document_path.name}")
        
        # Clear current data
        self.current_debug_data = None
        
        return html_file or debug_file
    
    def _save_debug_data(self) -> Path:
        """Save debug data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_debug_data.document_path.stem}_debug_{timestamp}.json"
        output_file = self.config.output_dir / filename
        
        # Convert to serializable format
        data = {
            "document_id": self.current_debug_data.document_id,
            "document_path": str(self.current_debug_data.document_path),
            "start_time": self.current_debug_data.start_time.isoformat(),
            "end_time": self.current_debug_data.end_time.isoformat() if self.current_debug_data.end_time else None,
            "timings": {
                "parsing": self.current_debug_data.parsing_time,
                "vlm_processing": self.current_debug_data.vlm_processing_time,
                "chunking": self.current_debug_data.chunking_time,
                "total": self.current_debug_data.total_time
            },
            "stats": self.current_debug_data.stats,
            "errors": self.current_debug_data.errors,
            "warnings": self.current_debug_data.warnings,
            "segments": [self._serialize_segment_info(s) for s in self.current_debug_data.segments],
            "chunks": [self._serialize_chunk_info(c) for c in self.current_debug_data.chunks],
            "visual_elements": self.current_debug_data.visual_elements
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def _serialize_segment_info(self, seg_info: SegmentDebugInfo) -> Dict[str, Any]:
        """Serialize segment debug info"""
        return {
            "segment_index": seg_info.segment.segment_index,
            "page_number": seg_info.segment.page_number,
            "segment_type": seg_info.segment.segment_type,
            "content_preview": seg_info.segment.content[:self.config.max_content_preview],
            "content_length": len(seg_info.segment.content),
            "processing_time": seg_info.processing_time,
            "vlm_applied": seg_info.vlm_applied,
            "vlm_model": seg_info.vlm_model,
            "vlm_description": seg_info.vlm_description,
            "vlm_confidence": seg_info.vlm_confidence,
            "error": seg_info.error,
            "metadata": seg_info.metadata
        }
    
    def _serialize_chunk_info(self, chunk_info: ChunkDebugInfo) -> Dict[str, Any]:
        """Serialize chunk debug info"""
        return {
            "chunk_id": chunk_info.chunk.chunk_id,
            "chunk_type": chunk_info.chunk.chunk_type.value,
            "content_preview": chunk_info.chunk.content[:self.config.max_content_preview],
            "content_length": len(chunk_info.chunk.content),
            "token_count": chunk_info.chunk.token_count,
            "source_segments": chunk_info.source_segments,
            "processing_time": chunk_info.processing_time,
            "context_inheritance_applied": chunk_info.context_inheritance_applied,
            "inherited_from": chunk_info.inherited_from,
            "error": chunk_info.error,
            "metadata": chunk_info.metadata
        }
    
    def _generate_html_report(self) -> Path:
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_debug_data.document_path.stem}_analysis_{timestamp}.html"
        output_file = self.config.output_dir / filename
        
        html_content = self._create_html_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def _create_html_report(self) -> str:
        """Create the HTML report content"""
        # This will be implemented in the next part
        # For now, return a placeholder
        return self._generate_html_template()
    
    def _generate_html_template(self) -> str:
        """Generate the HTML template with debug data"""
        doc_name = self.current_debug_data.document_path.name
        
        # Calculate statistics
        vlm_segments = sum(1 for s in self.current_debug_data.segments if s.vlm_applied)
        avg_confidence = sum(s.vlm_confidence for s in self.current_debug_data.segments if s.vlm_confidence) / vlm_segments if vlm_segments > 0 else 0
        
        html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Analysis - {doc_name}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Pipeline Analysis Report</h1>
        <p>{doc_name} - {self.current_debug_data.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <!-- Summary Section -->
        <div class="section">
            <h2>📈 Processing Summary</h2>
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
                    <p><strong>Processed Segments:</strong> {vlm_segments}</p>
                    <p><strong>Average Confidence:</strong> {avg_confidence:.2%}</p>
                    <p><strong>Models Used:</strong> {', '.join(set(s.vlm_model for s in self.current_debug_data.segments if s.vlm_model))}</p>
                </div>
                
                <div class="stat-card">
                    <h3>⚠️ Issues</h3>
                    <p><strong>Errors:</strong> {len(self.current_debug_data.errors)}</p>
                    <p><strong>Warnings:</strong> {len(self.current_debug_data.warnings)}</p>
                </div>
            </div>
        </div>
        
        <!-- Pipeline Flow Visualization -->
        <div class="section">
            <h2>🔄 Pipeline Flow</h2>
            <div class="pipeline-flow">
                <div class="pipeline-step completed">
                    <div class="step-icon">📄</div>
                    <div class="step-name">PDF Input</div>
                    <div class="step-time">{self.current_debug_data.parsing_time:.1f}s</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step completed">
                    <div class="step-icon">🤖</div>
                    <div class="step-name">SmolDocling</div>
                    <div class="step-segments">{self.current_debug_data.stats.get('total_segments', 0)} segments</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step {'completed' if vlm_segments > 0 else 'skipped'}">
                    <div class="step-icon">👁️</div>
                    <div class="step-name">VLM Analysis</div>
                    <div class="step-time">{self.current_debug_data.vlm_processing_time:.1f}s</div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step completed">
                    <div class="step-icon">📦</div>
                    <div class="step-name">Chunking</div>
                    <div class="step-chunks">{self.current_debug_data.stats.get('total_chunks', 0)} chunks</div>
                </div>
            </div>
        </div>
        
        <!-- Segments Analysis -->
        <div class="section">
            <h2>📝 Segments Analysis</h2>
            <div class="filter-controls">
                <button onclick="filterSegments('all')" class="filter-btn active">All Segments</button>
                <button onclick="filterSegments('vlm')" class="filter-btn">With VLM</button>
                <button onclick="filterSegments('text')" class="filter-btn">Text Only</button>
                <button onclick="filterSegments('visual')" class="filter-btn">Visual Elements</button>
            </div>
            <div id="segments-container" class="segments-grid">
                {self._generate_segments_html()}
            </div>
        </div>
        
        <!-- Chunks Analysis -->
        <div class="section">
            <h2>📦 Chunks Analysis</h2>
            <div class="chunks-grid">
                {self._generate_chunks_html()}
            </div>
        </div>
        
        <!-- Errors and Warnings -->
        {self._generate_issues_html()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>"""
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report"""
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
            max-width: 1400px;
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
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .stat-card h3 {
            margin-top: 0;
            color: #667eea;
        }
        
        .pipeline-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 2rem 0;
            padding: 2rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .pipeline-step {
            text-align: center;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 120px;
        }
        
        .pipeline-step.completed {
            border: 2px solid #28a745;
        }
        
        .pipeline-step.skipped {
            border: 2px solid #ffc107;
            opacity: 0.7;
        }
        
        .step-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .pipeline-arrow {
            font-size: 2rem;
            color: #667eea;
            margin: 0 1rem;
        }
        
        .filter-controls {
            margin-bottom: 1.5rem;
        }
        
        .filter-btn {
            background: #e9ecef;
            border: none;
            padding: 0.5rem 1rem;
            margin-right: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .filter-btn:hover {
            background: #dee2e6;
        }
        
        .filter-btn.active {
            background: #667eea;
            color: white;
        }
        
        .segments-grid, .chunks-grid {
            display: grid;
            gap: 1rem;
        }
        
        .segment-card, .chunk-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .segment-card.has-vlm {
            border-left: 4px solid #28a745;
        }
        
        .content-preview {
            background: white;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .vlm-description {
            background: #d4edda;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 4px;
            border-left: 3px solid #28a745;
        }
        
        .metadata {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #666;
        }
        
        .error-section {
            background: #f8d7da;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            margin-top: 1rem;
        }
        
        .warning-section {
            background: #fff3cd;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin-top: 1rem;
        }
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactivity"""
        return """
        function filterSegments(type) {
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            const segments = document.querySelectorAll('.segment-card');
            segments.forEach(segment => {
                if (type === 'all') {
                    segment.style.display = 'block';
                } else if (type === 'vlm' && segment.classList.contains('has-vlm')) {
                    segment.style.display = 'block';
                } else if (type === 'text' && segment.dataset.type === 'text') {
                    segment.style.display = 'block';
                } else if (type === 'visual' && segment.dataset.hasVisual === 'true') {
                    segment.style.display = 'block';
                } else {
                    segment.style.display = 'none';
                }
            });
        }
        
        function toggleContent(elementId) {
            const element = document.getElementById(elementId);
            element.style.display = element.style.display === 'none' ? 'block' : 'none';
        }
        """
    
    def _generate_segments_html(self) -> str:
        """Generate HTML for segments section"""
        html_parts = []
        
        for seg_info in self.current_debug_data.segments[:50]:  # Limit to first 50
            vlm_class = "has-vlm" if seg_info.vlm_applied else ""
            has_visual = bool(seg_info.segment.visual_references) if hasattr(seg_info.segment, 'visual_references') else False
            
            html_parts.append(f"""
            <div class="segment-card {vlm_class}" data-type="{seg_info.segment.segment_type}" 
                 data-has-visual="{str(has_visual).lower()}">
                <h4>Segment #{seg_info.segment.segment_index}</h4>
                <div class="metadata">
                    <span>📄 Page {seg_info.segment.page_number}</span>
                    <span>📝 Type: {seg_info.segment.segment_type}</span>
                    <span>⏱️ {seg_info.processing_time:.3f}s</span>
                </div>
                
                <div class="content-preview">
{seg_info.segment.content[:self.config.max_content_preview]}{'...' if len(seg_info.segment.content) > self.config.max_content_preview else ''}
                </div>
                
                {self._generate_vlm_section(seg_info) if seg_info.vlm_applied else ''}
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_vlm_section(self, seg_info: SegmentDebugInfo) -> str:
        """Generate VLM section for a segment"""
        return f"""
        <div class="vlm-description">
            <strong>🤖 VLM Analysis ({seg_info.vlm_model})</strong>
            <div style="margin-top: 0.5rem;">{seg_info.vlm_description}</div>
            <div style="margin-top: 0.5rem; color: #666;">
                Confidence: {seg_info.vlm_confidence:.2%}
            </div>
        </div>
        """
    
    def _generate_chunks_html(self) -> str:
        """Generate HTML for chunks section"""
        html_parts = []
        
        for chunk_info in self.current_debug_data.chunks[:30]:  # Limit to first 30
            html_parts.append(f"""
            <div class="chunk-card">
                <h4>Chunk {chunk_info.chunk.chunk_id}</h4>
                <div class="metadata">
                    <span>📦 Type: {chunk_info.chunk.chunk_type.value}</span>
                    <span>🔢 Tokens: {chunk_info.chunk.token_count}</span>
                    <span>📄 Sources: {len(chunk_info.source_segments)} segments</span>
                </div>
                
                <div class="content-preview">
{chunk_info.chunk.content[:self.config.max_content_preview]}{'...' if len(chunk_info.chunk.content) > self.config.max_content_preview else ''}
                </div>
                
                {self._generate_context_section(chunk_info) if chunk_info.context_inheritance_applied else ''}
            </div>
            """)
        
        return '\n'.join(html_parts)
    
    def _generate_context_section(self, chunk_info: ChunkDebugInfo) -> str:
        """Generate context inheritance section"""
        return f"""
        <div style="background: #e7f3ff; padding: 1rem; margin-top: 1rem; border-radius: 4px;">
            <strong>🔗 Inherited Context</strong>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                {chunk_info.chunk.inherited_context[:200]}...
            </div>
        </div>
        """
    
    def _generate_issues_html(self) -> str:
        """Generate HTML for errors and warnings"""
        if not self.current_debug_data.errors and not self.current_debug_data.warnings:
            return ""
            
        html = '<div class="section"><h2>⚠️ Issues Detected</h2>'
        
        if self.current_debug_data.errors:
            html += '<div class="error-section"><h3>❌ Errors</h3>'
            for error in self.current_debug_data.errors:
                html += f"""
                <div style="margin-bottom: 1rem;">
                    <strong>{error['stage']}</strong>: {error['error_type']}<br>
                    {error['message']}
                </div>
                """
            html += '</div>'
        
        if self.current_debug_data.warnings:
            html += '<div class="warning-section"><h3>⚠️ Warnings</h3>'
            for warning in self.current_debug_data.warnings:
                html += f"""
                <div style="margin-bottom: 0.5rem;">
                    <strong>{warning['stage']}</strong>: {warning['message']}
                </div>
                """
            html += '</div>'
        
        html += '</div>'
        return html


def create_debugger(debug_level: str = "basic", generate_report: bool = True) -> PipelineDebugger:
    """Factory function to create a debugger with common settings"""
    config = PipelineDebugConfig(
        debug_level=DebugLevel(debug_level),
        generate_html_report=generate_report,
        track_segments=True,
        track_chunks=True,
        track_vlm_descriptions=True,
        save_intermediate_results=debug_level in ["detailed", "full"],
        include_images=debug_level == "full"
    )
    return PipelineDebugger(config)