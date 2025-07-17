#!/usr/bin/env python3
"""
Generate HTML comparison using local SmolDocling implementation
"""

import sys
import asyncio
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import fitz  # PyMuPDF
from PIL import Image

from core.parsers import ParserFactory
from core.parsers.implementations.pdf import HybridPDFParser
from core.vlm.batch_processor import BatchDocumentProcessor, BatchProcessingConfig
from core.content_chunker import ContentChunker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentComparisonGenerator:
    """Generate HTML comparison of parsing pipeline stages"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Process document through complete pipeline"""
        logger.info(f"Processing {pdf_path.name}...")
        
        # Use HybridPDFParser which doesn't require external services
        parser = HybridPDFParser(enable_vlm=False)
        document = await parser.parse(pdf_path)
        
        # 2. Run VLM analysis if visual elements exist
        vlm_results = []
        if document.visual_elements:
            logger.info(f"Running VLM analysis on {len(document.visual_elements)} visual elements...")
            
            config = BatchProcessingConfig(
                batch_size=10,
                use_two_stage_processing=True,
                confidence_threshold=0.7,
                save_intermediate_results=True
            )
            
            processor = BatchDocumentProcessor(config)
            batch_results = await processor.process_documents([pdf_path])
            
            if batch_results and batch_results[0].visual_results:
                vlm_results = batch_results[0].visual_results
        
        # 3. Create chunks with VLM-enhanced content
        chunker = ContentChunker({
            "chunking": {
                "strategies": {
                    "pdf": {
                        "max_tokens": 500,
                        "min_tokens": 100,
                        "overlap_tokens": 50,
                        "respect_boundaries": True
                    }
                },
                "enable_context_inheritance": True,
                "context_inheritance": {
                    "enabled": True,
                    "max_context_tokens": 300
                }
            }
        })
        
        # Enhance document with VLM results
        if vlm_results:
            self._enhance_document_with_vlm(document, vlm_results)
        
        chunking_result = await chunker.chunk_document(document)
        
        # 4. Extract page images
        page_images = self._extract_page_images(pdf_path)
        
        return {
            "document": document,
            "vlm_results": vlm_results,
            "chunks": chunking_result.chunks,
            "page_images": page_images,
            "stats": {
                "total_segments": len(document.segments),
                "total_visual_elements": len(document.visual_elements),
                "total_chunks": len(chunking_result.chunks),
                "vlm_analyzed": len(vlm_results)
            }
        }
    
    def _enhance_document_with_vlm(self, document, vlm_results):
        """Enhance document segments with VLM analysis results"""
        # Create mapping of visual elements to VLM results
        vlm_map = {}
        for result in vlm_results:
            if hasattr(result, 'element_hash'):
                vlm_map[result.element_hash] = result
        
        # Update visual elements with VLM descriptions
        for visual in document.visual_elements:
            if visual.content_hash in vlm_map:
                vlm_result = vlm_map[visual.content_hash]
                visual.description = vlm_result.description
                visual.metadata["vlm_analysis"] = {
                    "model": vlm_result.model,
                    "confidence": vlm_result.confidence,
                    "extracted_data": vlm_result.extracted_data
                }
    
    def _extract_page_images(self, pdf_path: Path, dpi: int = 150) -> Dict[int, str]:
        """Extract page images as base64 strings"""
        page_images = {}
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                
                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                page_images[page_num + 1] = f"data:image/png;base64,{img_base64}"
                
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error extracting page images: {e}")
            
        return page_images
    
    def generate_comparison_html(self, pdf_path: Path, results: Dict[str, Any]) -> Path:
        """Generate HTML comparison report"""
        document = results["document"]
        chunks = results["chunks"]
        page_images = results["page_images"]
        vlm_results = results["vlm_results"]
        
        # Group segments and chunks by page
        segments_by_page = {}
        for segment in document.segments:
            page = segment.page_number
            if page not in segments_by_page:
                segments_by_page[page] = []
            segments_by_page[page].append(segment)
        
        chunks_by_page = {}
        for chunk in chunks:
            # Find which page this chunk primarily belongs to
            if hasattr(chunk, 'metadata') and 'page_numbers' in chunk.metadata:
                pages = chunk.metadata['page_numbers']
                primary_page = pages[0] if pages else 1
            else:
                primary_page = 1
                
            if primary_page not in chunks_by_page:
                chunks_by_page[primary_page] = []
            chunks_by_page[primary_page].append(chunk)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Comparison: {pdf_path.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
        }}
        
        .stat {{
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
        }}
        
        .page-comparison {{
            margin: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .page-header {{
            background: #f8f9fa;
            padding: 1rem;
            border-bottom: 1px solid #e9ecef;
            font-weight: bold;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 2rem;
            padding: 2rem;
        }}
        
        .column {{
            min-height: 400px;
        }}
        
        .column-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .original-page {{
            text-align: center;
        }}
        
        .original-page img {{
            max-width: 100%;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }}
        
        .segment {{
            background: #f8f9fa;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }}
        
        .segment-type {{
            font-size: 0.8rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }}
        
        .visual-element {{
            background: #e3f2fd;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            border-left: 3px solid #2196f3;
        }}
        
        .vlm-description {{
            background: #f3e5f5;
            padding: 0.5rem;
            margin-top: 0.5rem;
            border-radius: 4px;
            font-style: italic;
        }}
        
        .chunk {{
            background: #e8f5e9;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border-left: 3px solid #4caf50;
        }}
        
        .chunk-metadata {{
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 0.5rem;
        }}
        
        .context-info {{
            background: #fff3cd;
            padding: 0.5rem;
            margin-top: 0.5rem;
            border-radius: 4px;
            font-size: 0.9rem;
        }}
        
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
        }}
        
        .no-content {{
            color: #6c757d;
            font-style: italic;
            text-align: center;
            padding: 2rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Comparison: {pdf_path.name}</h1>
        <div class="stats">
            <div class="stat">📄 {results['stats']['total_segments']} Segments</div>
            <div class="stat">🖼️ {results['stats']['total_visual_elements']} Visual Elements</div>
            <div class="stat">🤖 {results['stats']['vlm_analyzed']} VLM Analyzed</div>
            <div class="stat">📦 {results['stats']['total_chunks']} Chunks</div>
        </div>
    </div>
"""
        
        # Generate comparison for each page
        for page_num in sorted(set(list(segments_by_page.keys()) + list(page_images.keys()))):
            html_content += f"""
    <div class="page-comparison">
        <div class="page-header">
            Seite {page_num}
        </div>
        <div class="comparison-grid">
            <div class="column">
                <div class="column-title">🖼️ Original PDF</div>
                <div class="original-page">
"""
            
            if page_num in page_images:
                html_content += f'<img src="{page_images[page_num]}" alt="Page {page_num}">'
            else:
                html_content += '<div class="no-content">Keine Seitenvorschau verfügbar</div>'
                
            html_content += """
                </div>
            </div>
            
            <div class="column">
                <div class="column-title">📝 Extrahierte Segmente</div>
"""
            
            # Add segments for this page
            if page_num in segments_by_page:
                for segment in segments_by_page[page_num]:
                    segment_type = getattr(segment, 'segment_type', 'text')
                    html_content += f"""
                <div class="segment">
                    <div class="segment-type">Type: {segment_type} | Index: {segment.segment_index}</div>
                    <pre>{self._escape_html(segment.content[:500])}</pre>
"""
                    
                    # Add visual elements if any
                    page_visuals = [v for v in document.visual_elements if v.page_number == page_num]
                    for visual in page_visuals:
                        vlm_desc = visual.metadata.get('vlm_analysis', {})
                        html_content += f"""
                    <div class="visual-element">
                        <strong>{visual.element_type.value.title()}</strong>
                        {f'<div class="vlm-description">VLM: {visual.description}</div>' if visual.description else ''}
                        {f'<div>Confidence: {vlm_desc.get("confidence", 0):.0%}</div>' if vlm_desc else ''}
                    </div>
"""
                    
                    html_content += "</div>"
            else:
                html_content += '<div class="no-content">Keine Segmente auf dieser Seite</div>'
                
            html_content += """
            </div>
            
            <div class="column">
                <div class="column-title">📦 Contextual Chunks</div>
"""
            
            # Add chunks for this page
            if page_num in chunks_by_page:
                for i, chunk in enumerate(chunks_by_page[page_num]):
                    html_content += f"""
                <div class="chunk">
                    <div class="segment-type">Chunk {chunk.chunk_id}</div>
                    <pre>{self._escape_html(chunk.content[:500])}</pre>
                    <div class="chunk-metadata">
                        Tokens: {chunk.token_count} | Type: {chunk.chunk_type.value}
                    </div>
"""
                    
                    # Add context information
                    if hasattr(chunk, 'inherited_context') and chunk.inherited_context:
                        html_content += f"""
                    <div class="context-info">
                        <strong>Context:</strong> {self._escape_html(str(chunk.inherited_context)[:200])}
                    </div>
"""
                    
                    html_content += "</div>"
            else:
                html_content += '<div class="no-content">Keine Chunks auf dieser Seite</div>'
                
            html_content += """
            </div>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML
        output_file = self.output_dir / f"{pdf_path.stem}_comparison.html"
        output_file.write_text(html_content, encoding='utf-8')
        
        return output_file
    
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


async def main():
    """Run segment comparison for BMW documents"""
    
    # Find BMW documents
    input_dir = Path("data/input")
    bmw_files = list(input_dir.glob("Preview_BMW*.pdf"))
    
    if not bmw_files:
        logger.error("No Preview_BMW*.pdf files found in data/input/")
        return
    
    logger.info(f"Found {len(bmw_files)} BMW documents")
    
    # Create output directory
    output_dir = Path("tests/debugging/segment_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each document
    generator = SegmentComparisonGenerator(output_dir)
    
    for pdf_file in bmw_files:
        try:
            logger.info(f"\nProcessing {pdf_file.name}...")
            
            # Process through pipeline
            results = await generator.process_document(pdf_file)
            
            # Generate HTML comparison
            html_file = generator.generate_comparison_html(pdf_file, results)
            
            logger.info(f"✅ Generated comparison: {html_file}")
            
            # Also save intermediate results as JSON
            json_file = output_dir / f"{pdf_file.stem}_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "stats": results["stats"],
                    "segments_sample": [s.to_dict() for s in results["document"].segments[:5]],
                    "chunks_sample": [c.to_dict() for c in results["chunks"][:5]],
                    "vlm_results_count": len(results["vlm_results"])
                }, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())