"""
vLLM SmolDocling client for advanced PDF parsing (GPU Workload 1)
Modernized with standardized BaseModelClient architecture
"""

import base64
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx

from datetime import datetime
from pydantic import BaseModel

from core.clients.base import BaseModelClient, BatchProcessingMixin
from plugins.parsers.base_parser import Document, Segment, DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


# Data Models
class SmolDoclingConfig(BaseModel):
    """Configuration for SmolDocling processing"""
    max_pages: int = 100
    extract_tables: bool = True
    extract_images: bool = True
    extract_formulas: bool = True
    preserve_layout: bool = True
    output_format: str = "structured"
    gpu_optimization: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300


class TableData(BaseModel):
    """Extracted table data structure"""
    caption: Optional[str] = None
    headers: List[str] = []
    rows: List[List[str]] = []
    page_number: int
    bbox: Optional[Dict[str, float]] = None


class ImageData(BaseModel):
    """Extracted image data structure"""
    caption: Optional[str] = None
    description: Optional[str] = None
    page_number: int
    bbox: Optional[Dict[str, float]] = None
    image_type: str = "figure"


class FormulaData(BaseModel):
    """Extracted formula data structure"""
    latex: Optional[str] = None
    mathml: Optional[str] = None
    description: Optional[str] = None
    page_number: int
    bbox: Optional[Dict[str, float]] = None


class SmolDoclingPage(BaseModel):
    """Parsed page data from SmolDocling"""
    page_number: int
    text: str
    tables: List[TableData] = []
    images: List[ImageData] = []
    formulas: List[FormulaData] = []
    layout_info: Dict[str, Any] = {}
    confidence_score: float = 0.0


class SmolDoclingResult(BaseModel):
    """Complete SmolDocling parsing result"""
    pages: List[SmolDoclingPage]
    metadata: Dict[str, Any] = {}
    processing_time_seconds: float
    model_version: str
    total_pages: int
    success: bool
    error_message: Optional[str] = None


class SmolDoclingRequest(BaseModel):
    """Request-Format für SmolDocling"""
    pdf_path: Path
    config: Optional[SmolDoclingConfig] = None


class VLLMSmolDoclingClient(BaseModelClient[SmolDoclingRequest, SmolDoclingResult, SmolDoclingConfig], 
                            BatchProcessingMixin):
    """
    Modernisierter vLLM SmolDocling Client
    
    Handles advanced PDF parsing including:
    - Complex document layouts
    - Table extraction with structure preservation
    - Image analysis and caption generation
    - Mathematical formula recognition
    - Multi-column text processing
    
    Now with:
    - Automatic retry logic
    - Standardized health checks
    - Built-in metrics
    - Batch processing support
    - Unified error handling
    """
    
    def __init__(self, config: Optional[SmolDoclingConfig] = None):
        """Initialize with backward compatibility"""
        # For backward compatibility - if config passed directly
        if config and not isinstance(config, str):
            super().__init__("vllm", config=config)
        else:
            super().__init__("vllm")
    
    def _get_default_config(self) -> SmolDoclingConfig:
        """Standard-Konfiguration für SmolDocling"""
        return SmolDoclingConfig()
    
    async def _process_internal(self, request: SmolDoclingRequest) -> SmolDoclingResult:
        """
        Interne PDF-Verarbeitung mit SmolDocling
        
        Args:
            request: PDF-Pfad und optionale Config
            
        Returns:
            SmolDocling Parsing-Ergebnis
        """
        # Verwende request-spezifische Config oder Default
        config = request.config or self.config
        
        # Lese PDF-Datei
        pdf_path = request.pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Bereite Request für vLLM vor
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        vllm_request = {
            "pdf_base64": base64.b64encode(pdf_content).decode('utf-8'),
            "config": {
                "max_pages": config.max_pages,
                "extract_tables": config.extract_tables,
                "extract_images": config.extract_images,
                "extract_formulas": config.extract_formulas,
                "preserve_layout": config.preserve_layout,
                "output_format": config.output_format,
                "gpu_optimization": config.gpu_optimization,
                "batch_size": config.batch_size
            }
        }
        
        # Sende Request an vLLM
        response = await self.client.post(
            f"{self.endpoint}/v1/parse",
            json=vllm_request
        )
        response.raise_for_status()
        
        # Parse Response
        result_data = response.json()
        
        # Konvertiere zu SmolDoclingResult
        pages = []
        for page_data in result_data.get("pages", []):
            page = SmolDoclingPage(
                page_number=page_data["page_number"],
                text=page_data["text"],
                tables=[TableData(**t) for t in page_data.get("tables", [])],
                images=[ImageData(**i) for i in page_data.get("images", [])],
                formulas=[FormulaData(**f) for f in page_data.get("formulas", [])],
                layout_info=page_data.get("layout_info", {}),
                confidence_score=page_data.get("confidence_score", 0.0)
            )
            pages.append(page)
        
        return SmolDoclingResult(
            pages=pages,
            metadata=result_data.get("metadata", {}),
            processing_time_seconds=result_data.get("processing_time", 0.0),
            model_version=result_data.get("model_version", "unknown"),
            total_pages=len(pages),
            success=True
        )
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """SmolDocling-spezifischer Health Check"""
        response = await self.client.get(f"{self.endpoint}/health")
        response.raise_for_status()
        
        health_data = response.json()
        
        # Extrahiere relevante Informationen
        return {
            "model_loaded": health_data.get("model_loaded", False),
            "model_name": health_data.get("model_name", "SmolDocling"),
            "gpu_memory_used": health_data.get("gpu_memory_used", 0),
            "gpu_memory_total": health_data.get("gpu_memory_total", 0),
            "queue_size": health_data.get("queue_size", 0),
            "version": health_data.get("version", "unknown")
        }
    
    async def parse_pdf(self, pdf_path: Path, config: Optional[SmolDoclingConfig] = None) -> Document:
        """
        Convenience-Methode für direktes PDF-Parsing
        
        Args:
            pdf_path: Pfad zur PDF-Datei
            config: Optionale Config-Überschreibung
            
        Returns:
            Geparster Document
        """
        # Erstelle Request
        request = SmolDoclingRequest(pdf_path=pdf_path, config=config)
        
        # Verarbeite mit BaseModelClient (inkl. Retry-Logik!)
        result = await self.process(request)
        
        # Konvertiere zu Document
        return self._convert_to_document(pdf_path, result)
    
    async def parse_multiple_pdfs(self, pdf_paths: List[Path]) -> List[Document]:
        """
        Parse mehrere PDFs parallel mit Batch-Processing
        
        Args:
            pdf_paths: Liste von PDF-Pfaden
            
        Returns:
            Liste von Documents
        """
        # Erstelle Requests
        requests = [SmolDoclingRequest(pdf_path=path) for path in pdf_paths]
        
        # Nutze BatchProcessingMixin
        results = await self.process_batch(
            requests, 
            batch_size=5,  # 5 PDFs pro Batch
            concurrent_batches=2  # 2 parallele Batches
        )
        
        # Konvertiere Ergebnisse
        documents = []
        for path, result in zip(pdf_paths, results):
            if result:  # Skip failed results
                documents.append(self._convert_to_document(path, result))
        
        return documents
    
    def _convert_to_document(self, pdf_path: Path, result: SmolDoclingResult) -> Document:
        """Konvertiere SmolDoclingResult zu Document"""
        segments = []
        
        for page in result.pages:
            # Text-Segmente
            if page.text.strip():
                segments.append(Segment(
                    content=page.text,
                    segment_type="text",
                    metadata={
                        "page": page.page_number,
                        "confidence": page.confidence_score
                    }
                ))
            
            # Tabellen-Segmente
            for table in page.tables:
                segments.append(Segment(
                    content=self._format_table(table),
                    segment_type="table",
                    metadata={
                        "page": page.page_number,
                        "caption": table.caption,
                        "headers": table.headers,
                        "rows": table.rows
                    }
                ))
            
            # Bild-Beschreibungen
            for image in page.images:
                if image.description:
                    segments.append(Segment(
                        content=image.description,
                        segment_type="image_description",
                        metadata={
                            "page": page.page_number,
                            "caption": image.caption,
                            "image_type": image.image_type
                        }
                    ))
        
        # Erstelle Document
        metadata = DocumentMetadata(
            title=pdf_path.stem,
            author=result.metadata.get("author"),
            created_date=None,  # Could parse from PDF metadata
            page_count=result.total_pages,
            document_type=DocumentType.PDF
        )
        
        return Document(
            document_id=f"smoldocling_{pdf_path.stem}_{hash(pdf_path)}",
            source_path=str(pdf_path),
            document_type=DocumentType.PDF,
            metadata=metadata,
            segments=segments,
            processing_timestamp=datetime.now().isoformat()
        )
    
    def _format_table(self, table: TableData) -> str:
        """Formatiere Tabelle als Text"""
        lines = []
        if table.caption:
            lines.append(f"Table: {table.caption}")
        
        # Headers
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * (len(" | ".join(table.headers))))
        
        # Rows
        for row in table.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)


# Beispiel-Nutzung:
async def example_usage():
    """Zeigt die Vorteile der neuen Architektur"""
    
    # Client mit automatischer Config aus System
    async with VLLMSmolDoclingClient("vllm") as client:
        
        # 1. Health Check (standardisiert!)
        health = await client.health_check()
        print(f"Service Status: {health.status}")
        print(f"Response Time: {health.response_time_ms}ms")
        
        # 2. Warte bis Service bereit
        ready = await client.wait_until_ready(max_attempts=10)
        if not ready:
            print("Service nicht bereit!")
            return
        
        # 3. Parse einzelne PDF (mit Auto-Retry!)
        doc = await client.parse_pdf(Path("test.pdf"))
        print(f"Parsed {len(doc.segments)} segments")
        
        # 4. Parse mehrere PDFs parallel
        pdf_files = Path("data/input").glob("*.pdf")
        documents = await client.parse_multiple_pdfs(list(pdf_files))
        print(f"Parsed {len(documents)} documents")
        
        # 5. Metriken abrufen
        metrics = client.get_metrics()
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Success Rate: {metrics.successful_requests / metrics.total_requests * 100:.1f}%")
        print(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")