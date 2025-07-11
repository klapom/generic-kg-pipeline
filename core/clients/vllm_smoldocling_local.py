"""
Local vLLM SmolDocling Client

High-performance PDF parsing using local vLLM SmolDocling model.
Supports multimodal document processing with optimized batch operations.
"""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from pdf2image import convert_from_path

from core.vllm.base_client import BaseVLLMClient, InferenceRequest, InferenceResult
from core.vllm.model_manager import ModelConfig, SamplingConfig
from plugins.parsers.base_parser import Document, Segment, DocumentMetadata, DocumentType, ParseError, VisualElement

logger = logging.getLogger(__name__)


@dataclass 
class SmolDoclingPage:
    """Parsed page data from SmolDocling"""
    page_number: int
    text: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    formulas: List[Dict[str, Any]]
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


class VLLMSmolDoclingClient(BaseVLLMClient):
    """
    Local vLLM SmolDocling Client
    
    Provides high-performance PDF parsing using local vLLM model with:
    - Multimodal PDF processing
    - Batch page processing
    - Structure-aware parsing
    - Visual element extraction
    """
    
    MODEL_NAME = "ds4sd/SmolDocling-256M-preview"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.12,  # SmolDocling needs only ~500MB, allow 5GB total
        max_pages: int = 100,
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_formulas: bool = True,
        preserve_layout: bool = True,
        auto_load: bool = False
    ):
        # Configure for better performance and quality
        import os
        os.environ['HF_HUB_OFFLINE'] = '0'  # Allow model downloads if needed
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_DATASETS_OFFLINE'] = '0'
        # Use XFORMERS backend for V0 engine which produces better results
        os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
        os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATES'] = '1'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'  # Allow longer max_tokens
        # Create model configuration optimized for SmolDocling with performance settings
        model_config = ModelConfig(
            model_name=self.MODEL_NAME,
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,  # SmolDocling max supported length
            trust_remote_code=False,  # Don't use remote code
            enforce_eager=False,  # Enable CUDA graphs for maximum performance
            limit_mm_per_prompt={"image": 1},  # Critical for multimodal processing
            dtype="bfloat16"  # Use bfloat16 for better quality
        )
        
        # Sampling configuration optimized for SmolDocling
        sampling_config = SamplingConfig(
            temperature=0.0,  # Deterministic for consistent results
            max_tokens=8192,  # Full model capacity
            top_k=50,  # Limit vocab sampling
            top_p=0.95,  # Nucleus sampling
            stop=["</s>", "<|endoftext|>"]  # Standard stop tokens only
        )
        # Don't use repetition penalties - they cause issues with SmolDocling
        self.repetition_penalty = 1.0  # No penalty
        self.presence_penalty = 0.0  # No penalty
        self.frequency_penalty = 0.0  # No penalty
        
        # SmolDocling-specific configuration
        self.max_pages = max_pages
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.extract_formulas = extract_formulas
        self.preserve_layout = preserve_layout
        
        super().__init__(
            model_id="smoldocling",
            model_config=model_config,
            sampling_config=sampling_config,
            auto_load=auto_load
        )
        
        logger.info(f"Initialized VLLMSmolDoclingClient:")
        logger.info(f"  - Max pages: {max_pages}")
        logger.info(f"  - GPU memory utilization: {gpu_memory_utilization:.1%} (~{gpu_memory_utilization * 44:.1f}GB on 44GB GPU)")
        logger.info(f"  - Model: {self.MODEL_NAME}")
        logger.info(f"  - Data type: bfloat16 (better quality)")
        logger.info(f"  - Enforce eager: False (CUDA graphs ENABLED for speed)")
        logger.info(f"  - Max model length: 8192")
        logger.info(f"  - Max tokens: 8192")
        logger.info(f"  - Temperature: 0.0 (deterministic)")
        logger.info(f"  - Repetition penalty: 1.0 (disabled - causes issues with SmolDocling)")
        logger.info(f"  - Image quality: 300 DPI PNG (optimized for SmolDocling)")
        logger.info(f"  - Multimodal limit: {model_config.limit_mm_per_prompt}")
    
    def get_warmup_prompts(self) -> List[str]:
        """SmolDocling-specific warmup prompts"""
        return [
            "Process this document page and extract structured information.",
            "Parse this PDF page with tables and images."
        ]
    
    def _get_input_size(self, args, kwargs) -> int:
        """Extract input size for performance tracking"""
        if args and hasattr(args[0], 'prompts'):
            request = args[0]
            if isinstance(request.prompts, list):
                return len(request.prompts)
        return 1
    
    def _get_output_size(self, result) -> int:
        """Extract output size for performance tracking"""
        if isinstance(result, InferenceResult) and result.outputs:
            return len(result.outputs)
        return 0
    
    def _build_parsing_prompt(self) -> str:
        """Build the parsing prompt for SmolDocling using correct chat template"""
        # Use the exact prompt from the reference implementation
        return "Convert page to Docling."
    
    def _prepare_page_for_processing(self, page_image: Image.Image) -> str:
        """Prepare page image for vLLM processing"""
        try:
            # Convert to RGB if necessary
            if page_image.mode in ('RGBA', 'LA', 'P'):
                page_image = page_image.convert('RGB')
            
            # Resize if too large - SmolDocling might work better with smaller images
            max_size = 1536  # Reduced from 2048
            if max(page_image.size) > max_size:
                page_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from original to {page_image.size}")
            
            # For vLLM, we should pass the PIL Image directly, not base64
            return page_image
            
        except Exception as e:
            logger.error(f"Failed to prepare page image: {e}")
            raise ParseError(f"Image preparation failed: {str(e)}")
    
    def clean_doctags(self, raw: str) -> str:
        """Remove docling-specific tags from raw output"""
        import re
        return re.sub(r"<loc_\d+>|</?end_of_utterance>", "", raw)
    
    def is_complex_layout(self, raw_content: str, parsed_data: dict) -> dict:
        """
        Detect if SmolDocling interpreted the page as a single complex image
        Returns detection results with confidence scores
        """
        import re
        
        # Count elements
        picture_count = raw_content.count('<picture>')
        table_count = raw_content.count('<otsl>') + raw_content.count('<table>')
        text_count = raw_content.count('<text>')
        
        # Get actual parsed counts
        parsed_pictures = len(parsed_data.get('images', []))
        parsed_tables = len(parsed_data.get('tables', []))
        parsed_text_blocks = len(parsed_data.get('text_blocks', []))
        total_text_length = len(parsed_data.get('text', ''))
        
        # Detection logic with multiple criteria
        is_single_picture = picture_count == 1 and parsed_pictures == 1
        has_no_tables = table_count == 0 and parsed_tables == 0
        has_minimal_text = text_count < 2 and parsed_text_blocks < 2
        has_short_text = total_text_length < 100  # Less than 100 chars
        
        # Check if the single picture covers most of the page
        covers_full_page = False
        if parsed_pictures == 1 and 'images' in parsed_data and parsed_data['images']:
            img = parsed_data['images'][0]
            if 'content' in img:
                # Check if location tags indicate full page coverage
                loc_match = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', img['content'])
                if loc_match:
                    x1, y1, x2, y2 = map(int, loc_match.groups())
                    # If image covers >80% of page dimensions (0-500 scale)
                    width_coverage = (x2 - x1) / 500
                    height_coverage = (y2 - y1) / 500
                    covers_full_page = width_coverage > 0.8 and height_coverage > 0.8
        
        # Complex layout detection
        is_complex = (
            is_single_picture and 
            has_no_tables and 
            has_minimal_text and 
            has_short_text and
            covers_full_page
        )
        
        detection_info = {
            'is_complex_layout': is_complex,
            'detection_details': {
                'picture_count': picture_count,
                'table_count': table_count,
                'text_count': text_count,
                'parsed_pictures': parsed_pictures,
                'parsed_tables': parsed_tables,
                'parsed_text_blocks': parsed_text_blocks,
                'total_text_length': total_text_length,
                'is_single_picture': is_single_picture,
                'has_no_tables': has_no_tables,
                'has_minimal_text': has_minimal_text,
                'has_short_text': has_short_text,
                'covers_full_page': covers_full_page
            },
            'confidence': 0.9 if is_complex else 0.1
        }
        
        if is_complex:
            logger.warning(f"🚨 Complex layout detected! Single picture covering full page with no text/tables extracted")
            logger.debug(f"Detection details: {detection_info['detection_details']}")
        
        return detection_info
    
    def parse_model_output(self, output: Any) -> Dict[str, Any]:
        """Parse SmolDocling DocTags output format"""
        try:
            # Extract text from vLLM output
            if hasattr(output, 'outputs') and output.outputs:
                content = output.outputs[0].text
            elif hasattr(output, 'text'):
                content = output.text
            else:
                content = str(output)
            
            # Clean up the content - remove leading/trailing whitespace
            content = content.strip()
            
            logger.debug(f"Raw DocTags output (first 1000 chars): {content[:1000]}")
            
            # Extract DocTags structured content
            import re
            
            # DocTags format uses <loc_x1><loc_y1><loc_x2><loc_y2> tags
            # Extract text content with location info
            text_blocks = []
            # Updated pattern to match actual DocTags format
            text_with_loc = re.findall(r"<text><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>(.+?)</text>", content, flags=re.DOTALL)
            for x1, y1, x2, y2, text in text_with_loc:
                text_blocks.append({
                    "text": text.strip(),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
            
            # Also extract text that might not have location tags (older format)
            text_only = re.findall(r"<text>(?!<loc_)(.+?)</text>", content, flags=re.DOTALL)
            for text in text_only:
                if not any(text.strip() == tb["text"] for tb in text_blocks):
                    text_blocks.append({"text": text.strip(), "bbox": None})
            
            # Extract titles/headers
            titles = re.findall(r"<title>(.+?)</title>", content, flags=re.DOTALL)
            section_headers = re.findall(r"<section_header>(.+?)</section_header>", content, flags=re.DOTALL)
            
            # Extract tables with OTSL format
            tables = []
            table_matches = re.findall(r"<otsl>(.+?)</otsl>", content, flags=re.DOTALL)
            for table_content in table_matches:
                tables.append({
                    "content": table_content.strip(),
                    "format": "otsl"
                })
            
            # Extract pictures with captions
            pictures = []
            picture_matches = re.findall(r"<picture>(.+?)</picture>", content, flags=re.DOTALL)
            for pic in picture_matches:
                caption_match = re.search(r"<caption>(.+?)</caption>", pic)
                pictures.append({
                    "content": pic.strip(),
                    "caption": caption_match.group(1).strip() if caption_match else ""
                })
            
            # Extract formulas
            formulas = []
            formula_matches = re.findall(r"<formula>(.+?)</formula>", content, flags=re.DOTALL)
            for formula in formula_matches:
                formulas.append({
                    "latex": formula.strip(),
                    "type": "formula"
                })
            
            # Extract code blocks
            code_blocks = []
            code_matches = re.findall(r"<code>(.+?)</code>", content, flags=re.DOTALL)
            for code in code_matches:
                code_blocks.append({
                    "content": code.strip(),
                    "language": "unknown"
                })
            
            # Extract list items (including location tags)
            list_items = re.findall(r"<list_item>(?:<loc_\d+>)*(.+?)</list_item>", content, flags=re.DOTALL)
            # Clean location tags from list items
            cleaned_list_items = []
            for item in list_items:
                clean_item = re.sub(r'<loc_\d+>', '', item).strip()
                if clean_item:
                    cleaned_list_items.append(clean_item)
            
            # Combine all text content
            all_text = []
            
            # Add titles and headers first
            all_text.extend(titles)
            all_text.extend(section_headers)
            
            # Add text blocks
            for tb in text_blocks:
                all_text.append(tb["text"])
            
            # Add list items
            all_text.extend([f"• {item}" for item in cleaned_list_items])
            
            # Add code blocks
            for code in code_blocks:
                all_text.append(f"[CODE]\n{code['content']}")
            
            # If no structured content found, try to extract any readable text
            if not all_text:
                # Remove tags and extract remaining text
                clean_content = re.sub(r"<[^>]+>", " ", content)
                clean_content = re.sub(r"\s+", " ", clean_content).strip()
                if clean_content:
                    all_text = [clean_content]
            
            # Log extracted elements for debugging
            logger.debug(f"Extracted {len(text_blocks)} text blocks, {len(titles)} titles, "
                        f"{len(tables)} tables, {len(pictures)} pictures, {len(formulas)} formulas, "
                        f"{len(code_blocks)} code blocks")
            
            # Return structured data
            parsed_data = {
                "text": "\n\n".join(all_text) if all_text else "",
                "tables": tables,
                "images": pictures,
                "formulas": formulas,
                "code_blocks": code_blocks,
                "text_blocks": text_blocks,
                "titles": titles,
                "layout_info": {
                    "raw_content": content,
                    "doctags_elements": {
                        "text_count": len(text_blocks),
                        "table_count": len(tables),
                        "picture_count": len(pictures),
                        "formula_count": len(formulas),
                        "code_count": len(code_blocks)
                    }
                },
                "confidence_score": 1.0 if all_text else 0.5
            }
            
            # Check for complex layout
            detection_info = self.is_complex_layout(content, parsed_data)
            parsed_data['layout_info']['complex_layout_detection'] = detection_info
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to parse SmolDocling DocTags output: {e}")
            logger.debug(f"Raw output: {content[:500] if 'content' in locals() else 'No content'}...")
            # Return raw content as fallback
            return {
                "text": content if 'content' in locals() else "",
                "tables": [],
                "images": [],
                "formulas": [],
                "code_blocks": [],
                "text_blocks": [],
                "titles": [],
                "layout_info": {"error": str(e)},
                "confidence_score": 0.0
            }
    
    def process_pdf_page(self, page_image: Image.Image, page_number: int) -> SmolDoclingPage:
        """Process a single PDF page"""
        try:
            # Prepare image (convert to RGB, resize if needed)
            processed_image = self._prepare_page_for_processing(page_image)
            
            # Use the correct vLLM multimodal format with exact prompt
            PROMPT_TEXT = self._build_parsing_prompt()
            
            # Exact chat template from reference implementation
            chat_template = f"<|im_start|>User:<image>{PROMPT_TEXT}<|im_end|><|im_start|>Assistant:"
            
            # Correct vLLM multimodal input format with PIL Image directly
            multimodal_input = {
                "prompt": chat_template,  # String, not list
                "multi_modal_data": {"image": processed_image}  # PIL Image directly
            }
            
            # Create vLLM SamplingParams directly for better control
            from vllm import SamplingParams
            
            # Use vLLM SamplingParams directly - no repetition penalties for SmolDocling
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=8192,  # Maximum supported by SmolDocling
                # Don't use repetition penalties - they cause issues with SmolDocling
                stop=["</s>", "<|endoftext|>", "</doctag>", "<|im_end|>"],  # Standard stop tokens
                skip_special_tokens=True,  # Skip special tokens in output
                include_stop_str_in_output=False  # Don't include stop string
            )
            
            # Generate using model_manager which handles multimodal correctly
            if not self.model_manager.is_model_loaded(self.model_id):
                raise ParseError("SmolDocling model not loaded")
            
            # Use model_manager's generate method with multimodal input and direct sampling params
            # Get the model directly for more control
            model = self.model_manager._models[self.model_id]
            
            # Generate directly with vLLM model
            result = model.generate(multimodal_input, sampling_params)
            
            # Handle direct vLLM output format
            if hasattr(result, 'outputs'):
                outputs = result.outputs
            elif isinstance(result, list):
                outputs = result
            else:
                outputs = [result]
            
            if not outputs:
                raise ParseError("No output from SmolDocling model")
            
            # Log raw V2T output
            raw_output = outputs[0]
            logger.info(f"\n{'='*80}")
            logger.info(f"🔍 RAW V2T OUTPUT for page {page_number}:")
            logger.info(f"{'='*80}")
            if hasattr(raw_output, 'outputs') and len(raw_output.outputs) > 0:
                # Extract text from completion output
                raw_text = raw_output.outputs[0].text
                logger.info("--- START RAW V2T TEXT ---")
                logger.info(raw_text)
                logger.info("--- END RAW V2T TEXT ---")
            elif hasattr(raw_output, 'text'):
                logger.info("--- START RAW V2T TEXT ---")
                logger.info(raw_output.text)
                logger.info("--- END RAW V2T TEXT ---")
            else:
                logger.info(f"Raw output type: {type(raw_output)}")
                logger.info(f"Raw output: {str(raw_output)[:500]}...")
            
            # Parse the output
            parsed_data = self.parse_model_output(raw_output)
            
            # Store the raw vLLM response in layout_info
            if hasattr(raw_output, 'outputs') and len(raw_output.outputs) > 0:
                raw_text = raw_output.outputs[0].text
            elif hasattr(raw_output, 'text'):
                raw_text = raw_output.text
            else:
                raw_text = str(raw_output)
            
            # Add raw response to layout_info
            parsed_data['layout_info']['vllm_response'] = raw_text
            
            # Create SmolDoclingPage
            page = SmolDoclingPage(
                page_number=page_number,
                text=parsed_data.get("text", ""),
                tables=parsed_data.get("tables", []),
                images=parsed_data.get("images", []),
                formulas=parsed_data.get("formulas", []),
                layout_info=parsed_data.get("layout_info", {}),
                confidence_score=parsed_data.get("confidence_score", 0.0)
            )
            
            # Check if complex layout was detected
            complex_detection = page.layout_info.get('complex_layout_detection', {})
            if complex_detection.get('is_complex_layout', False):
                logger.warning(f"📑 LOCAL vLLM - Page {page_number} COMPLEX LAYOUT DETECTED: "
                            f"{len(page.images)} images, "
                            f"{len(page.text)} chars extracted, "
                            f"needs_fallback=True")
            else:
                logger.info(f"📑 LOCAL vLLM - Processed page {page_number}: "
                            f"{len(page.tables)} tables, "
                            f"{len(page.images)} images, "
                            f"{len(page.formulas)} formulas, "
                            f"{len(page.text)} chars extracted")
            
            return page
            
        except Exception as e:
            logger.error(f"Failed to process page {page_number}: {e}")
            # Return empty page on error
            return SmolDoclingPage(
                page_number=page_number,
                text=f"Error processing page {page_number}: {str(e)}",
                tables=[],
                images=[],
                formulas=[],
                layout_info={},
                confidence_score=0.0
            )
    
    def parse_pdf(self, pdf_path: Path) -> SmolDoclingResult:
        """
        Parse a PDF document using local vLLM SmolDocling
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            SmolDocling parsing result
        """
        start_time = datetime.now()
        
        if not pdf_path.exists():
            raise ParseError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ParseError(f"Expected PDF file, got: {pdf_path.suffix}")
        
        logger.info(f"🚀 Starting LOCAL vLLM SmolDocling PDF parsing: {pdf_path.name}")
        
        try:
            # Ensure model is loaded (skip warmup)
            logger.info("🔧 Loading SmolDocling model in LOCAL vLLM...")
            if not self.model_manager.is_model_loaded(self.model_id):
                success = self.model_manager.load_model(self.model_id)
                if not success:
                    raise ParseError("Failed to load SmolDocling model")
                # Skip warmup for faster startup
                logger.info("⚡ Skipping warmup for faster processing")
            logger.info("✅ SmolDocling model ready in LOCAL vLLM")
            
            # Convert PDF to images with higher quality to avoid SmolDocling issues
            logger.info("🖼️ Converting PDF pages to images with high quality...")
            page_images = convert_from_path(
                str(pdf_path),
                first_page=1,
                last_page=min(self.max_pages, 100),  # Safety limit
                dpi=300,  # Higher DPI as recommended for SmolDocling
                fmt='PNG'  # PNG format for better quality
            )
            
            if not page_images:
                raise ParseError("Failed to convert PDF to images")
            
            logger.info(f"✅ Converted {len(page_images)} PDF pages to images")
            logger.info(f"🔧 Processing {len(page_images)} pages with LOCAL vLLM SmolDocling...")
            
            # Process pages
            pages = []
            for i, page_image in enumerate(page_images):
                page_number = i + 1
                logger.info(f"📄 Processing page {page_number}/{len(page_images)} with LOCAL vLLM...")
                
                page_result = self.process_pdf_page(page_image, page_number)
                pages.append(page_result)
            
            # Create metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = {
                "title": pdf_path.stem,
                "source_file": str(pdf_path),
                "total_pages": len(pages),
                "processing_time": processing_time,
                "model": self.MODEL_NAME,
                "extraction_settings": {
                    "extract_tables": self.extract_tables,
                    "extract_images": self.extract_images,
                    "extract_formulas": self.extract_formulas,
                    "preserve_layout": self.preserve_layout
                }
            }
            
            result = SmolDoclingResult(
                pages=pages,
                metadata=metadata,
                processing_time_seconds=processing_time,
                model_version=self.MODEL_NAME,
                total_pages=len(pages),
                success=True
            )
            
            logger.info(f"✅ PDF parsing completed: {pdf_path.name}")
            logger.info(f"   Pages processed: {len(pages)}")
            logger.info(f"   Processing time: {processing_time:.1f}s")
            logger.info(f"   Tables found: {sum(len(p.tables) for p in pages)}")
            logger.info(f"   Images found: {sum(len(p.images) for p in pages)}")
            logger.info(f"   Formulas found: {sum(len(p.formulas) for p in pages)}")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"PDF parsing failed: {pdf_path.name} - {str(e)}")
            
            return SmolDoclingResult(
                pages=[],
                metadata={"error": str(e)},
                processing_time_seconds=processing_time,
                model_version=self.MODEL_NAME,
                total_pages=0,
                success=False,
                error_message=str(e)
            )
    
    def convert_to_document(self, result: SmolDoclingResult, pdf_path: Path) -> Document:
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
        
        # Combine all text content and create segments
        full_text_parts = []
        segments = []
        visual_elements = []
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
                    metadata={
                        "confidence_score": page.confidence_score,
                        "layout_info": page.layout_info
                    }
                ))
                segment_index += 1
            
            # Add table content
            for table in page.tables:
                table_text = self._table_to_text(table)
                if table_text:
                    full_text_parts.append(table_text)
                    segments.append(Segment(
                        content=table_text,
                        page_number=page.page_number,
                        segment_index=segment_index,
                        segment_type="table",
                        metadata={
                            "format": table.get("format", "otsl"),
                            "raw_content": table.get("content")
                        }
                    ))
                    segment_index += 1
            
            # Add image descriptions and visual elements
            for image in page.images:
                if image.get("description"):
                    image_text = f"[Image: {image.get('caption', 'Untitled')}] {image['description']}"
                    full_text_parts.append(image_text)
                    segments.append(Segment(
                        content=image_text,
                        page_number=page.page_number,
                        segment_index=segment_index,
                        segment_type="image_caption",
                        metadata={
                            "caption": image.get("caption"),
                            "image_type": image.get("image_type", "figure"),
                            "bbox": image.get("bbox")
                        }
                    ))
                    segment_index += 1
                
                # Add to visual elements
                visual_elements.append(VisualElement(
                    element_type=image.get("image_type", "figure"),
                    content=image.get("description", ""),
                    page_number=page.page_number,
                    bbox=image.get("bbox"),
                    metadata={
                        "caption": image.get("caption"),
                        "extracted_by": "SmolDocling"
                    }
                ))
            
            # Add formula descriptions
            for formula in page.formulas:
                if formula.get("description") or formula.get("latex"):
                    formula_text = f"[Formula] {formula.get('description', formula.get('latex', ''))}"
                    full_text_parts.append(formula_text)
                    segments.append(Segment(
                        content=formula_text,
                        page_number=page.page_number,
                        segment_index=segment_index,
                        segment_type="formula",
                        metadata={
                            "latex": formula.get("latex"),
                            "bbox": formula.get("bbox")
                        }
                    ))
                    segment_index += 1
        
        # Create document metadata
        stat = pdf_path.stat()
        metadata = DocumentMetadata(
            title=result.metadata.get("title", pdf_path.stem),
            page_count=result.total_pages,
            file_size=stat.st_size,
            file_path=pdf_path,
            document_type=DocumentType.PDF,
            created_date=datetime.fromtimestamp(stat.st_ctime),
            custom_metadata={
                "smoldocling": {
                    "model_version": result.model_version,
                    "processing_time_seconds": result.processing_time_seconds,
                    "tables_count": sum(len(p.tables) for p in result.pages),
                    "images_count": sum(len(p.images) for p in result.pages),
                    "formulas_count": sum(len(p.formulas) for p in result.pages),
                    "total_segments": len(segments),
                    "extraction_settings": result.metadata.get("extraction_settings", {})
                }
            }
        )
        
        full_text = "\n\n".join(full_text_parts)
        
        return Document(
            content=full_text,
            segments=segments,
            metadata=metadata,
            visual_elements=visual_elements,
            raw_data=result
        )
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to readable text format"""
        # For DocTags OTSL format, return the content directly
        if table.get("format") == "otsl":
            return table.get("content", "")
        
        # Fallback for other formats
        return table.get("content", "")
    
    def batch_parse_pdfs(self, pdf_paths: List[Path]) -> List[SmolDoclingResult]:
        """
        Parse multiple PDFs in batch for better performance
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of SmolDocling parsing results
        """
        if not pdf_paths:
            return []
        
        logger.info(f"Starting batch PDF parsing: {len(pdf_paths)} files")
        
        # Ensure model is loaded once for the entire batch
        if not self.ensure_model_loaded():
            logger.error("Failed to load SmolDocling model for batch processing")
            return [SmolDoclingResult(
                pages=[],
                metadata={"error": "Model loading failed"},
                processing_time_seconds=0.0,
                model_version=self.MODEL_NAME,
                total_pages=0,
                success=False,
                error_message="Model loading failed"
            ) for _ in pdf_paths]
        
        results = []
        start_time = datetime.now()
        
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_path.name}")
            
            try:
                result = self.parse_pdf(pdf_path)
                results.append(result)
                
                if result.success:
                    logger.info(f"✅ {pdf_path.name}: {result.total_pages} pages")
                else:
                    logger.error(f"❌ {pdf_path.name}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ {pdf_path.name}: {str(e)}")
                results.append(SmolDoclingResult(
                    pages=[],
                    metadata={"error": str(e)},
                    processing_time_seconds=0.0,
                    model_version=self.MODEL_NAME,
                    total_pages=0,
                    success=False,
                    error_message=str(e)
                ))
        
        total_time = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results if r.success)
        
        logger.info(f"✅ Batch PDF parsing completed:")
        logger.info(f"   Total files: {len(pdf_paths)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {len(pdf_paths) - successful}")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Average per file: {total_time/len(pdf_paths):.1f}s")
        
        return results