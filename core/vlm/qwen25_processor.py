#!/usr/bin/env python3
"""
Qwen2.5-VL Processor for Visual Element Analysis

This processor replaces the TwoStageVLMProcessor with a single, more efficient
Qwen2.5-VL based processor that handles both individual visual elements and
page-level context analysis.
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ..parsers.interfaces import VisualElement, VisualElementType
from ..clients.transformers_qwen25_vl_client import TransformersQwen25VLClient

logger = logging.getLogger(__name__)


class VisualAnalysisResult:
    """Result from visual analysis"""
    def __init__(self, 
                 success: bool,
                 description: Optional[str] = None,
                 structured_data: Optional[Dict[str, Any]] = None,
                 confidence: Optional[float] = None,
                 ocr_text: Optional[str] = None,
                 error_message: Optional[str] = None):
        self.success = success
        self.description = description
        self.structured_data = structured_data
        self.confidence = confidence
        self.ocr_text = ocr_text
        self.error_message = error_message


class PageContext:
    """Context information for a document page"""
    def __init__(self,
                 page_type: str,
                 main_topic: Optional[str] = None,
                 key_information: Optional[List[str]] = None,
                 element_relationships: Optional[Dict[str, Any]] = None):
        self.page_type = page_type
        self.main_topic = main_topic
        self.key_information = key_information or []
        self.element_relationships = element_relationships or {}


class Qwen25VLMProcessor:
    """Single-stage VLM processor using only Qwen2.5-VL"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Qwen2.5-VL processor
        
        Args:
            config: Configuration dictionary with:
                - temperature: Model temperature (default: 0.2)
                - max_new_tokens: Maximum new tokens (default: 512)
                - batch_size: Batch size for processing (default: 4)
                - enable_page_context: Enable page-level analysis (default: True)
                - enable_structured_parsing: Enable JSON parsing (default: True)
        """
        self.config = config
        
        # Initialize Qwen2.5-VL client
        self.client = TransformersQwen25VLClient(
            temperature=config.get('temperature', 0.2),
            max_new_tokens=config.get('max_new_tokens', 512)
        )
        
        self.batch_size = config.get('batch_size', 4)
        self.enable_page_context = config.get('enable_page_context', True)
        self.enable_structured_parsing = config.get('enable_structured_parsing', True)
        
        logger.info(f"Initialized Qwen25VLMProcessor with config: {config}")
    
    async def process_visual_elements(self, 
                                    visual_elements: List[VisualElement],
                                    page_contexts: Optional[Dict[int, PageContext]] = None
                                    ) -> List[VisualAnalysisResult]:
        """
        Process visual elements with optional page context
        
        Args:
            visual_elements: List of visual elements to analyze
            page_contexts: Optional dictionary of page number to PageContext
            
        Returns:
            List of VisualAnalysisResult objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(visual_elements), self.batch_size):
            batch = visual_elements[i:i + self.batch_size]
            batch_results = await self._process_batch(batch, page_contexts)
            results.extend(batch_results)
        
        return results
    
    async def _process_batch(self,
                           batch: List[VisualElement],
                           page_contexts: Optional[Dict[int, PageContext]] = None
                           ) -> List[VisualAnalysisResult]:
        """Process a batch of visual elements"""
        tasks = []
        
        for ve in batch:
            # Get page context if available
            page_context = None
            if page_contexts and ve.page_or_slide in page_contexts:
                page_context = page_contexts[ve.page_or_slide]
            
            # Create analysis task
            task = self._analyze_visual_element(ve, page_context)
            tasks.append(task)
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(VisualAnalysisResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _analyze_visual_element(self,
                                    visual_element: VisualElement,
                                    page_context: Optional[PageContext] = None
                                    ) -> VisualAnalysisResult:
        """Analyze a single visual element"""
        try:
            if not visual_element.raw_data:
                return VisualAnalysisResult(
                    success=False,
                    error_message="No image data available"
                )
            
            # Build prompt based on element type and context
            prompt = self._build_prompt(visual_element, page_context)
            
            # Analyze with Qwen2.5-VL
            # Note: TransformersQwen25VLClient doesn't take prompt directly
            # It builds its own prompt based on element_type and analysis_focus
            result = self.client.analyze_visual(
                image_data=visual_element.raw_data,
                element_type=visual_element.element_type,
                analysis_focus="comprehensive",
                document_context={"custom_prompt": prompt} if page_context else None
            )
            
            if not result.success:
                return VisualAnalysisResult(
                    success=False,
                    error_message=result.error_message
                )
            
            # Parse structured response if enabled
            structured_data = None
            if self.enable_structured_parsing:
                structured_data = self.parse_structured_response(result.description)
            
            return VisualAnalysisResult(
                success=True,
                description=result.description,
                structured_data=structured_data,
                confidence=result.confidence,
                ocr_text=result.ocr_text
            )
            
        except Exception as e:
            logger.error(f"Error analyzing visual element: {e}")
            return VisualAnalysisResult(
                success=False,
                error_message=str(e)
            )
    
    def _build_prompt(self,
                     visual_element: VisualElement,
                     page_context: Optional[PageContext] = None
                     ) -> str:
        """Build analysis prompt based on element type and context"""
        base_prompt = "Analyze this image and provide detailed information."
        
        # Element-type specific prompts
        if visual_element.element_type == VisualElementType.TABLE:
            base_prompt = """Analyze this table image. Extract:
1. Table structure (rows, columns)
2. Header information
3. All data values
4. Any footnotes or captions

If the table contains structured data, format your response as JSON."""
        
        elif visual_element.element_type == VisualElementType.CHART:
            base_prompt = """Analyze this chart/graph. Describe:
1. Chart type
2. Axes labels and units
3. Data series and values
4. Key insights or trends
5. Any legends or annotations"""
        
        elif visual_element.element_type == VisualElementType.DIAGRAM:
            base_prompt = """Analyze this diagram. Identify:
1. Diagram type and purpose
2. Main components and their relationships
3. Any text labels or annotations
4. Flow of information or process
5. Key technical details"""
        
        # Add page context if available
        if page_context:
            context_info = f"\n\nPage context: This appears on a {page_context.page_type} page"
            if page_context.main_topic:
                context_info += f" about {page_context.main_topic}"
            base_prompt += context_info
        
        return base_prompt
    
    def parse_structured_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse structured data from VLM responses
        Handles both plain text and JSON responses
        """
        if not response:
            return None
        
        # Try to find JSON blocks in the response
        json_patterns = [
            r'\{[\s\S]*\}',  # Basic JSON object
            r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON block
            r'```\s*([\s\S]*?)\s*```'  # Generic code block
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                try:
                    # Clean up the match
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    # Try to parse as JSON
                    data = json.loads(match)
                    
                    # Validate it's a meaningful structure
                    if isinstance(data, dict) and len(data) > 0:
                        return {
                            'type': 'structured',
                            'format': 'json',
                            'data': data,
                            'source_text': response
                        }
                except json.JSONDecodeError:
                    continue
        
        # Try to extract structured information from text
        structured_info = self._extract_structured_from_text(response)
        if structured_info:
            return {
                'type': 'structured',
                'format': 'text_extracted',
                'data': structured_info,
                'source_text': response
            }
        
        # Fallback to plain text
        return {
            'type': 'text',
            'format': 'plain',
            'data': None,
            'source_text': response
        }
    
    def _extract_structured_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract structured information from plain text responses"""
        # Look for common patterns like "Model: X5", "Motor: B58", etc.
        patterns = {
            'model': r'(?:Model|Modell):\s*([^\n,]+)',
            'motor': r'(?:Motor|Engine):\s*([^\n,]+)',
            'power': r'(\d+)\s*(?:kW|PS|hp)',
            'torque': r'(\d+)\s*(?:Nm|lb-ft)',
            'table_rows': r'(?:Row|Zeile)\s*\d+:\s*([^\n]+)',
            'columns': r'(?:Column|Spalte)s?:\s*([^\n]+)'
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[key] = matches if len(matches) > 1 else matches[0]
        
        return extracted if extracted else None
    
    async def analyze_page_context(self, 
                                 page_image: bytes,
                                 page_num: int
                                 ) -> PageContext:
        """
        Analyze entire page for context
        
        Args:
            page_image: Rendered page as image bytes
            page_num: Page number
            
        Returns:
            PageContext object with page-level information
        """
        prompt = """Analyze this document page and provide:
1. Page type (title page, table of contents, content, technical specification, etc.)
2. Main topic or theme of the page
3. Key information points (as a list)
4. Relationships between visual and text elements

Format your response as JSON with keys: page_type, main_topic, key_information, element_relationships"""
        
        try:
            result = self.client.analyze_visual(
                image_data=page_image,
                analysis_focus="comprehensive",
                document_context={"page_context_prompt": prompt, "page_num": page_num}
            )
            
            if not result.success:
                logger.warning(f"Page context analysis failed for page {page_num}: {result.error_message}")
                return PageContext(page_type="unknown")
            
            # Parse the response
            parsed = self.parse_structured_response(result.description)
            
            if parsed and parsed.get('type') == 'structured' and parsed.get('data'):
                data = parsed['data']
                return PageContext(
                    page_type=data.get('page_type', 'content'),
                    main_topic=data.get('main_topic'),
                    key_information=data.get('key_information', []),
                    element_relationships=data.get('element_relationships', {})
                )
            else:
                # Fallback: try to extract page type from text
                page_type = 'content'  # default
                text_lower = result.description.lower()
                
                if 'title' in text_lower or 'deckblatt' in text_lower:
                    page_type = 'title'
                elif 'content' in text_lower or 'inhalt' in text_lower:
                    page_type = 'table_of_contents'
                elif 'technical' in text_lower or 'technisch' in text_lower:
                    page_type = 'technical_specification'
                
                return PageContext(page_type=page_type)
                
        except Exception as e:
            logger.error(f"Error analyzing page context: {e}")
            return PageContext(page_type="error")
    
    def enhance_with_context(self,
                           visual_element: VisualElement,
                           analysis_result: VisualAnalysisResult,
                           page_context: PageContext) -> None:
        """
        Enhance visual element with page context information
        
        This modifies the visual_element in-place, adding context metadata
        """
        if not page_context or not analysis_result.success:
            return
        
        # Add page context to analysis metadata
        if not visual_element.analysis_metadata:
            visual_element.analysis_metadata = {}
        
        visual_element.analysis_metadata['page_context'] = {
            'type': page_context.page_type,
            'main_topic': page_context.main_topic,
            'timestamp': datetime.now().isoformat()
        }
        
        # For technical specification pages, add special handling
        if page_context.page_type == "technical_specification":
            # Check if this is a motorization table
            if (visual_element.element_type == VisualElementType.TABLE and
                analysis_result.structured_data and
                any(key in str(analysis_result.structured_data).lower() 
                    for key in ['motor', 'modell', 'leistung', 'power'])):
                
                visual_element.analysis_metadata['table_type'] = 'motorization'
                visual_element.analysis_metadata['requires_structured_extraction'] = True
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.client, 'cleanup'):
            self.client.cleanup()
        logger.info("Qwen25VLMProcessor cleanup completed")