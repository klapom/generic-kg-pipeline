"""vLLM SmolDocling client for advanced PDF parsing (GPU Workload 1)"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel

from core.config import get_config
from plugins.parsers.base_parser import Document, Segment, DocumentMetadata, DocumentType, ParseError

logger = logging.getLogger(__name__)


class SmolDoclingConfig(BaseModel):
    """Configuration for SmolDocling processing"""
    max_pages: int = 100
    extract_tables: bool = True
    extract_images: bool = True
    extract_formulas: bool = True
    preserve_layout: bool = True
    output_format: str = "structured"  # structured, markdown, text
    gpu_optimization: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300


@dataclass
class TableData:
    """Extracted table data structure"""
    caption: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    bbox: Optional[Dict[str, float]] = None  # Bounding box coordinates


@dataclass
class ImageData:
    """Extracted image data structure"""
    caption: Optional[str]
    description: Optional[str]
    page_number: int
    bbox: Optional[Dict[str, float]] = None
    image_type: str = "figure"  # figure, diagram, chart, photo


@dataclass
class FormulaData:
    """Extracted formula data structure"""
    latex: Optional[str]
    mathml: Optional[str]
    description: Optional[str]
    page_number: int
    bbox: Optional[Dict[str, float]] = None


@dataclass
class SmolDoclingPage:
    """Parsed page data from SmolDocling"""
    page_number: int
    text: str
    tables: List[TableData]
    images: List[ImageData]
    formulas: List[FormulaData]
    layout_info: Dict[str, Any]
    confidence_score: float = 0.0


@dataclass
class SmolDoclingResult:
    """Complete SmolDocling parsing result"""
    pages: List[SmolDoclingPage]
    metadata: Dict[str, Any]
    processing_time_seconds: float
    model_version: str
    total_pages: int
    success: bool
    error_message: Optional[str] = None


class VLLMSmolDoclingClient:
    """
    Client for vLLM SmolDocling service (GPU Workload 1)
    
    Handles advanced PDF parsing including:
    - Complex document layouts
    - Table extraction with structure preservation
    - Image analysis and caption generation
    - Mathematical formula recognition
    - Multi-column text processing
    """
    
    def __init__(self, config: Optional[SmolDoclingConfig] = None):
        """Initialize the vLLM SmolDocling client"""
        self.config = config or SmolDoclingConfig()
        
        # Get endpoint from system config
        system_config = get_config()
        self.endpoint = system_config.parsing.pdf.vllm_endpoint
        
        # HTTP client with appropriate timeouts for GPU processing
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        logger.info(f"Initialized vLLM SmolDocling client: {self.endpoint}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the vLLM SmolDocling service is healthy
        
        Returns:
            Health status information
        """
        try:
            response = await self.client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            
            health_data = response.json()
            
            return {
                "status": "healthy",
                "endpoint": self.endpoint,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "model_info": health_data.get("model_info", {}),
                "gpu_info": health_data.get("gpu_info", {}),
                "last_check": datetime.now().isoformat()
            }
            
        except httpx.TimeoutException:
            logger.error("vLLM SmolDocling health check timed out")
            return {
                "status": "timeout",
                "endpoint": self.endpoint,
                "error": "Health check timed out",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"vLLM SmolDocling health check failed: {e}")
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def parse_pdf(
        self,
        pdf_path: Path,
        config: Optional[SmolDoclingConfig] = None
    ) -> SmolDoclingResult:
        """
        Parse a PDF document using vLLM SmolDocling
        
        Args:
            pdf_path: Path to the PDF file
            config: Optional configuration override
            
        Returns:
            SmolDocling parsing result
            
        Raises:
            ParseError: If parsing fails
        """
        start_time = datetime.now()
        config = config or self.config
        
        if not pdf_path.exists():
            raise ParseError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ParseError(f"Expected PDF file, got: {pdf_path.suffix}")
        
        try:
            # Read and encode PDF file
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "model": "smoldocling",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._build_parsing_prompt(config)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.1,
                "stream": False,
                "extra_body": {
                    "max_pages": config.max_pages,
                    "extract_tables": config.extract_tables,
                    "extract_images": config.extract_images,
                    "extract_formulas": config.extract_formulas,
                    "preserve_layout": config.preserve_layout,
                    "output_format": config.output_format,
                    "gpu_optimization": config.gpu_optimization
                }
            }
            
            logger.info(f"Sending PDF to vLLM SmolDocling: {pdf_path.name} ({len(pdf_content)} bytes)")
            
            # Send request to vLLM
            response = await self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            # Parse the response
            parsing_result = self._parse_vllm_response(result_data, start_time)
            
            logger.info(f"PDF parsing completed: {pdf_path.name} "
                       f"({parsing_result.total_pages} pages, "
                       f"{parsing_result.processing_time_seconds:.2f}s)")
            
            return parsing_result
            
        except httpx.TimeoutException:
            error_msg = f"PDF parsing timed out after {config.timeout_seconds}s: {pdf_path.name}"
            logger.error(error_msg)
            raise ParseError(error_msg)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"vLLM SmolDocling HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)
            raise ParseError(error_msg)
            
        except Exception as e:
            error_msg = f"PDF parsing failed: {pdf_path.name} - {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ParseError(error_msg)
    
    def _build_parsing_prompt(self, config: SmolDoclingConfig) -> str:
        """Build the parsing prompt for SmolDocling"""
        prompt_parts = [
            "Please parse this PDF document and extract structured information.",
            "Provide the results in JSON format with the following structure:",
            "",
            "```json",
            "{",
            "  \"pages\": [",
            "    {",
            "      \"page_number\": 1,",
            "      \"text\": \"extracted text content\",",
            "      \"tables\": [",
            "        {",
            "          \"caption\": \"table caption\",",
            "          \"headers\": [\"col1\", \"col2\"],",
            "          \"rows\": [[\"data1\", \"data2\"]],",
            "          \"page_number\": 1",
            "        }",
            "      ],",
            "      \"images\": [",
            "        {",
            "          \"caption\": \"image caption\",",
            "          \"description\": \"detailed description\",",
            "          \"page_number\": 1,",
            "          \"image_type\": \"figure\"",
            "        }",
            "      ],",
            "      \"formulas\": [",
            "        {",
            "          \"latex\": \"LaTeX representation\",",
            "          \"description\": \"formula description\",",
            "          \"page_number\": 1",
            "        }",
            "      ]",
            "    }",
            "  ],",
            "  \"metadata\": {",
            "    \"title\": \"document title\",",
            "    \"author\": \"document author\",",
            "    \"total_pages\": 10",
            "  }",
            "}",
            "```",
            ""
        ]
        
        if config.extract_tables:
            prompt_parts.append("- Extract all tables with their structure preserved")
        
        if config.extract_images:
            prompt_parts.append("- Analyze images and provide detailed descriptions")
        
        if config.extract_formulas:
            prompt_parts.append("- Extract mathematical formulas in LaTeX format")
        
        if config.preserve_layout:
            prompt_parts.append("- Preserve document layout and formatting information")
        
        prompt_parts.extend([
            "",
            f"Process up to {config.max_pages} pages.",
            "Focus on accuracy and completeness of extraction."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_vllm_response(
        self,
        response_data: Dict[str, Any],
        start_time: datetime
    ) -> SmolDoclingResult:
        """Parse the response from vLLM SmolDocling"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        try:
            # Extract the content from vLLM response
            choices = response_data.get("choices", [])
            if not choices:
                raise ValueError("No choices in vLLM response")
            
            content = choices[0].get("message", {}).get("content", "")
            
            # Parse JSON content
            if "```json" in content:
                # Extract JSON from markdown code block
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content
            
            parsed_data = json.loads(json_content)
            
            # Convert to SmolDocling data structures
            pages = []
            for page_data in parsed_data.get("pages", []):
                # Parse tables
                tables = []
                for table_data in page_data.get("tables", []):
                    tables.append(TableData(
                        caption=table_data.get("caption"),
                        headers=table_data.get("headers", []),
                        rows=table_data.get("rows", []),
                        page_number=table_data.get("page_number", page_data["page_number"]),
                        bbox=table_data.get("bbox")
                    ))
                
                # Parse images
                images = []
                for image_data in page_data.get("images", []):
                    images.append(ImageData(
                        caption=image_data.get("caption"),
                        description=image_data.get("description"),
                        page_number=image_data.get("page_number", page_data["page_number"]),
                        bbox=image_data.get("bbox"),
                        image_type=image_data.get("image_type", "figure")
                    ))
                
                # Parse formulas
                formulas = []
                for formula_data in page_data.get("formulas", []):
                    formulas.append(FormulaData(
                        latex=formula_data.get("latex"),
                        mathml=formula_data.get("mathml"),
                        description=formula_data.get("description"),
                        page_number=formula_data.get("page_number", page_data["page_number"]),
                        bbox=formula_data.get("bbox")
                    ))
                
                pages.append(SmolDoclingPage(
                    page_number=page_data["page_number"],
                    text=page_data.get("text", ""),
                    tables=tables,
                    images=images,
                    formulas=formulas,
                    layout_info=page_data.get("layout_info", {}),
                    confidence_score=page_data.get("confidence_score", 0.0)
                ))
            
            metadata = parsed_data.get("metadata", {})
            total_pages = metadata.get("total_pages", len(pages))
            
            # Get model version from response
            model_version = response_data.get("model", "smoldocling-unknown")
            
            return SmolDoclingResult(
                pages=pages,
                metadata=metadata,
                processing_time_seconds=processing_time,
                model_version=model_version,
                total_pages=total_pages,
                success=True
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse vLLM SmolDocling response: {e}")
            return SmolDoclingResult(
                pages=[],
                metadata={},
                processing_time_seconds=processing_time,
                model_version="unknown",
                total_pages=0,
                success=False,
                error_message=f"Response parsing failed: {str(e)}"
            )
    
    def convert_to_document(
        self,
        result: SmolDoclingResult,
        pdf_path: Path
    ) -> Document:
        """
        Convert SmolDocling result to standard Document format
        
        Args:
            result: SmolDocling parsing result
            pdf_path: Original PDF file path
            
        Returns:
            Document object compatible with the pipeline
        """
        if not result.success:
            raise ParseError(f"Cannot convert failed parsing result: {result.error_message}")
        
        # Combine all text content
        full_text_parts = []
        segments = []
        segment_index = 0
        
        for page in result.pages:
            # Add main text content
            if page.text:
                full_text_parts.append(page.text)
                segments.append(Segment(
                    content=page.text,
                    page_number=page.page_number,
                    segment_index=segment_index,
                    segment_type="text",
                    metadata={"confidence_score": page.confidence_score}
                ))
                segment_index += 1
            
            # Add table content
            for table in page.tables:
                table_text = self._table_to_text(table)
                if table_text:
                    full_text_parts.append(table_text)
                    segments.append(Segment(
                        content=table_text,
                        page_number=table.page_number,
                        segment_index=segment_index,
                        segment_type="table",
                        metadata={
                            "caption": table.caption,
                            "headers": table.headers,
                            "row_count": len(table.rows),
                            "column_count": len(table.headers),
                            "bbox": table.bbox
                        }
                    ))
                    segment_index += 1
            
            # Add image descriptions
            for image in page.images:
                if image.description:
                    image_text = f"[Image: {image.caption or 'Untitled'}] {image.description}"
                    full_text_parts.append(image_text)
                    segments.append(Segment(
                        content=image_text,
                        page_number=image.page_number,
                        segment_index=segment_index,
                        segment_type="image_caption",
                        metadata={
                            "caption": image.caption,
                            "image_type": image.image_type,
                            "bbox": image.bbox
                        }
                    ))
                    segment_index += 1
            
            # Add formula descriptions
            for formula in page.formulas:
                if formula.description or formula.latex:
                    formula_text = f"[Formula] {formula.description or formula.latex}"
                    full_text_parts.append(formula_text)
                    segments.append(Segment(
                        content=formula_text,
                        page_number=formula.page_number,
                        segment_index=segment_index,
                        segment_type="formula",
                        metadata={
                            "latex": formula.latex,
                            "mathml": formula.mathml,
                            "bbox": formula.bbox
                        }
                    ))
                    segment_index += 1
        
        # Create document metadata
        stat = pdf_path.stat()
        metadata = DocumentMetadata(
            title=result.metadata.get("title", pdf_path.stem),
            author=result.metadata.get("author"),
            page_count=result.total_pages,
            file_size=stat.st_size,
            file_path=pdf_path,
            document_type=DocumentType.PDF,
            custom_metadata={
                "vllm_smoldocling": {
                    "model_version": result.model_version,
                    "processing_time_seconds": result.processing_time_seconds,
                    "tables_count": sum(len(p.tables) for p in result.pages),
                    "images_count": sum(len(p.images) for p in result.pages),
                    "formulas_count": sum(len(p.formulas) for p in result.pages),
                    "total_segments": len(segments)
                }
            }
        )
        
        full_text = "\n\n".join(full_text_parts)
        
        return Document(
            content=full_text,
            segments=segments,
            metadata=metadata,
            raw_data=result
        )
    
    def _table_to_text(self, table: TableData) -> str:
        """Convert table data to readable text format"""
        if not table.rows:
            return table.caption or ""
        
        text_parts = []
        
        if table.caption:
            text_parts.append(f"Table: {table.caption}")
        
        # Add headers if available
        if table.headers:
            text_parts.append(" | ".join(table.headers))
            text_parts.append("-" * len(" | ".join(table.headers)))
        
        # Add rows
        for row in table.rows:
            text_parts.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(text_parts)
    
    async def batch_parse_pdfs(
        self,
        pdf_paths: List[Path],
        config: Optional[SmolDoclingConfig] = None
    ) -> List[SmolDoclingResult]:
        """
        Parse multiple PDFs in batch for better GPU utilization
        
        Args:
            pdf_paths: List of PDF file paths
            config: Optional configuration override
            
        Returns:
            List of SmolDocling parsing results
        """
        config = config or self.config
        
        # Process in batches to manage memory usage
        batch_size = config.batch_size
        results = []
        
        for i in range(0, len(pdf_paths), batch_size):
            batch = pdf_paths[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.parse_pdf(pdf_path, config)
                for pdf_path in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for pdf_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch parsing failed for {pdf_path}: {result}")
                        # Create failed result
                        results.append(SmolDoclingResult(
                            pages=[],
                            metadata={},
                            processing_time_seconds=0.0,
                            model_version="unknown",
                            total_pages=0,
                            success=False,
                            error_message=str(result)
                        ))
                    else:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add failed results for the entire batch
                for pdf_path in batch:
                    results.append(SmolDoclingResult(
                        pages=[],
                        metadata={},
                        processing_time_seconds=0.0,
                        model_version="unknown",
                        total_pages=0,
                        success=False,
                        error_message=f"Batch processing failed: {str(e)}"
                    ))
        
        logger.info(f"Batch parsing completed: {len(pdf_paths)} PDFs processed")
        return results