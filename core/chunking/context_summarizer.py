"""Context summarizer for generating context summaries using LLM"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.clients.hochschul_llm import HochschulLLMClient
from .chunk_models import ContextualChunk, ContextInheritance

logger = logging.getLogger(__name__)


class ContextSummaryResult:
    """Result of context summary generation"""
    
    def __init__(
        self,
        context_summary: str,
        main_task_result: Optional[str] = None,
        generation_time: float = 0.0,
        token_count: int = 0,
        quality_score: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        self.context_summary = context_summary
        self.main_task_result = main_task_result
        self.generation_time = generation_time
        self.token_count = token_count
        self.quality_score = quality_score
        self.success = success
        self.error_message = error_message


class ContextSummarizer:
    """
    Generates context summaries using LLM for context inheritance
    
    Uses dual-task prompting to generate both main task results and
    context summaries for inheritance to subsequent chunks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize context summarizer"""
        self.config = config
        self.context_config = config.get("context_inheritance", {})
        self.llm_config = self.context_config.get("llm", {})
        
        # Initialize LLM client
        self.llm_client = HochschulLLMClient()
        
        # Context generation settings
        self.max_context_tokens = self.context_config.get("max_context_tokens", 300)
        self.min_context_tokens = self.context_config.get("min_context_tokens", 50)
        self.context_quality_threshold = self.context_config.get("context_quality_threshold", 0.7)
        
        # Template settings
        self.template_config = config.get("templates", {})
        
        logger.info("Initialized ContextSummarizer")
    
    async def generate_context_summary(
        self,
        chunk: ContextualChunk,
        task_template: str,
        previous_context: Optional[str] = None
    ) -> ContextSummaryResult:
        """
        Generate context summary for inheritance to next chunks
        
        Args:
            chunk: The chunk to process and summarize
            task_template: Main task template for the chunk
            previous_context: Any inherited context from previous chunks
            
        Returns:
            ContextSummaryResult with context summary and main task result
        """
        start_time = datetime.now()
        
        try:
            # Build context generation prompt
            prompt = self._build_context_generation_prompt(
                chunk, 
                task_template, 
                previous_context
            )
            
            # Generate completion using LLM
            response = await self.llm_client.generate_completion(
                prompt=prompt,
                max_tokens=self.llm_config.get("max_tokens", 400),
                temperature=self.llm_config.get("temperature", 0.1),
                timeout=self.llm_config.get("timeout_seconds", 30)
            )
            
            if not response.success:
                return ContextSummaryResult(
                    context_summary="",
                    success=False,
                    error_message=f"LLM generation failed: {response.error_message}"
                )
            
            # Parse response to extract context summary and main result
            context_summary, main_result = self._parse_llm_response(response.content)
            
            # Validate context summary
            context_summary = self._validate_and_clean_context(context_summary, chunk)
            
            # Calculate quality score
            quality_score = self._calculate_context_quality(context_summary, chunk)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate token count
            token_count = self._estimate_tokens(context_summary)
            
            return ContextSummaryResult(
                context_summary=context_summary,
                main_task_result=main_result,
                generation_time=processing_time,
                token_count=token_count,
                quality_score=quality_score,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Context summary generation failed: {e}")
            
            return ContextSummaryResult(
                context_summary="",
                generation_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _build_context_generation_prompt(
        self,
        chunk: ContextualChunk,
        task_template: str,
        previous_context: Optional[str] = None
    ) -> str:
        """Build prompt for context summary generation"""
        
        # Base prompt structure
        prompt_parts = [
            "Du analysierst einen Textabschnitt und führst zwei Aufgaben aus:",
            "",
            f"HAUPTAUFGABE: {task_template}",
            "",
            "ZUSÄTZLICHE AUFGABE: Erstelle eine Kontextzusammenfassung für nachfolgende Textabschnitte.",
            "",
            "Die Kontextzusammenfassung soll:",
            "1. Kernthemen und wichtige Konzepte erfassen (2-3 Hauptpunkte)",
            "2. Relevante Definitionen und Erklärungen enthalten",
            "3. Wichtige Referenzen und Verweise festhalten",
            "4. Den thematischen 'roten Faden' bewahren",
            f"5. Zwischen {self.min_context_tokens} und {self.max_context_tokens} Tokens umfassen",
            "6. Prägnant und fokussiert bleiben",
            ""
        ]
        
        # Add previous context if available
        if previous_context:
            prompt_parts.extend([
                "VORHERIGER KONTEXT (von vorherigen Abschnitten):",
                previous_context,
                "",
                "WICHTIG: Aktualisiere und erweitere den vorherigen Kontext mit neuen Informationen aus dem aktuellen Textabschnitt.",
                ""
            ])
        
        # Add current text content
        enhanced_content = chunk.get_enhanced_content()
        prompt_parts.extend([
            "AKTUELLER TEXTABSCHNITT:",
            enhanced_content,
            ""
        ])
        
        # Add visual elements context if available
        if chunk.visual_elements:
            visual_context = self._format_visual_context(chunk.visual_elements)
            prompt_parts.extend([
                "VISUELLE ELEMENTE:",
                visual_context,
                "",
                "Berücksichtige diese visuellen Elemente in der Kontextzusammenfassung.",
                ""
            ])
        
        # Add output format specification
        prompt_parts.extend([
            "AUSGABEFORMAT:",
            "HAUPTAUFGABE ERGEBNIS:",
            "[Hier das Ergebnis der Hauptaufgabe]",
            "",
            "KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:",
            "[Kompakte Zusammenfassung der relevanten Kontextinformationen]",
            "",
            "WICHTIGE HINWEISE:",
            "- Halte die Kontextzusammenfassung prägnant und fokussiert",
            "- Verwende keine Füllwörter oder redundante Informationen",
            "- Konzentriere dich auf Informationen, die für nachfolgende Abschnitte relevant sind",
            "- Erwähne wichtige Definitionen und Konzepte explizit"
        ])
        
        return "\n".join(prompt_parts)
    
    def _format_visual_context(self, visual_elements: List[Any]) -> str:
        """Format visual elements for context prompt"""
        visual_descriptions = []
        
        for i, visual in enumerate(visual_elements):
            if visual.vlm_description:
                visual_descriptions.append(f"{i+1}. {visual.element_type.value.upper()}: {visual.vlm_description}")
            
            if visual.extracted_data:
                data_str = self._format_extracted_data(visual.extracted_data)
                visual_descriptions.append(f"   Daten: {data_str}")
        
        return "\n".join(visual_descriptions) if visual_descriptions else "Keine visuellen Elemente"
    
    def _format_extracted_data(self, data: Dict[str, Any]) -> str:
        """Format extracted data for display"""
        if not data:
            return "Keine Daten"
        
        # Handle different data types
        if "chart_type" in data:
            chart_type = data["chart_type"]
            data_points = data.get("data_points", {})
            return f"Chart ({chart_type}): {data_points}"
        elif "table_data" in data:
            return f"Tabelle: {data['table_data']}"
        elif "headers" in data and "rows" in data:
            headers = data["headers"]
            row_count = len(data.get("rows", []))
            return f"Tabelle: {headers} ({row_count} Zeilen)"
        else:
            # Generic formatting
            return str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
    
    def _parse_llm_response(self, response_content: str) -> Tuple[str, Optional[str]]:
        """Parse LLM response to extract context summary and main result"""
        
        # Look for context section markers
        context_markers = [
            "KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:",
            "CONTEXT FOR FOLLOWING SECTIONS:",
            "CONTEXT SUMMARY:",
            "KONTEXTZUSAMMENFASSUNG:",
            "ZUSAMMENFASSUNG:"
        ]
        
        # Look for main task result markers
        main_task_markers = [
            "HAUPTAUFGABE ERGEBNIS:",
            "MAIN TASK RESULT:",
            "ERGEBNIS:",
            "RESULT:"
        ]
        
        context_summary = ""
        main_result = None
        
        # Extract context summary
        for marker in context_markers:
            if marker in response_content:
                parts = response_content.split(marker, 1)
                if len(parts) > 1:
                    # Take everything after the marker until next section or end
                    context_part = parts[1]
                    
                    # Stop at next section marker
                    for other_marker in main_task_markers:
                        if other_marker in context_part:
                            context_part = context_part.split(other_marker)[0]
                            break
                    
                    context_summary = context_part.strip()
                    break
        
        # Extract main task result
        for marker in main_task_markers:
            if marker in response_content:
                parts = response_content.split(marker, 1)
                if len(parts) > 1:
                    # Take everything after the marker until context section
                    main_part = parts[1]
                    
                    # Stop at context section marker
                    for context_marker in context_markers:
                        if context_marker in main_part:
                            main_part = main_part.split(context_marker)[0]
                            break
                    
                    main_result = main_part.strip()
                    break
        
        # Fallback parsing if no markers found
        if not context_summary:
            # Try to extract from last paragraph
            paragraphs = response_content.split('\n\n')
            if paragraphs:
                context_summary = paragraphs[-1].strip()
        
        return context_summary, main_result
    
    def _validate_and_clean_context(self, context_summary: str, chunk: ContextualChunk) -> str:
        """Validate and clean context summary"""
        if not context_summary:
            return ""
        
        # Remove empty lines and excessive whitespace
        context_summary = re.sub(r'\n\s*\n', '\n', context_summary).strip()
        
        # Remove markdown formatting
        context_summary = re.sub(r'[*_`]', '', context_summary)
        
        # Remove section markers if they leaked through
        markers_to_remove = [
            "KONTEXT FÜR NACHFOLGENDE ABSCHNITTE:",
            "CONTEXT FOR FOLLOWING SECTIONS:",
            "HAUPTAUFGABE ERGEBNIS:",
            "MAIN TASK RESULT:"
        ]
        
        for marker in markers_to_remove:
            context_summary = context_summary.replace(marker, "").strip()
        
        # Ensure reasonable length
        token_count = self._estimate_tokens(context_summary)
        
        if token_count > self.max_context_tokens:
            # Truncate at sentence boundary
            sentences = context_summary.split('. ')
            truncated = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self._estimate_tokens(sentence)
                if current_tokens + sentence_tokens > self.max_context_tokens:
                    break
                truncated.append(sentence)
                current_tokens += sentence_tokens
            
            context_summary = '. '.join(truncated)
            if not context_summary.endswith('.'):
                context_summary += '.'
        
        # Ensure minimum length
        if token_count < self.min_context_tokens:
            logger.warning(f"Context summary too short ({token_count} tokens) for chunk {chunk.chunk_id}")
        
        return context_summary
    
    def _calculate_context_quality(self, context_summary: str, chunk: ContextualChunk) -> float:
        """Calculate quality score for context summary"""
        if not context_summary:
            return 0.0
        
        quality_score = 0.0
        
        # Check length appropriateness
        token_count = self._estimate_tokens(context_summary)
        if self.min_context_tokens <= token_count <= self.max_context_tokens:
            quality_score += 0.3
        
        # Check for key concepts (simplified)
        key_indicators = [
            "konzept", "definition", "wichtig", "hauptthema", "kernpunkt",
            "concept", "definition", "important", "main", "key",
            "thema", "inhalt", "aspekt", "punkt", "element"
        ]
        
        context_lower = context_summary.lower()
        concept_score = sum(1 for indicator in key_indicators if indicator in context_lower)
        quality_score += min(concept_score * 0.1, 0.3)
        
        # Check for structure (sentences, periods)
        sentence_count = len([s for s in context_summary.split('.') if s.strip()])
        if 2 <= sentence_count <= 5:
            quality_score += 0.2
        
        # Check for coherence (no repetition)
        words = context_summary.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness_ratio = len(unique_words) / len(words)
            quality_score += uniqueness_ratio * 0.2
        
        return min(quality_score, 1.0)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not text:
            return 0
        
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback approximation
            return len(text.split()) * 1.3
        except Exception:
            return len(text.split()) * 1.3
    
    async def process_context_inheritance(
        self,
        chunks: List[ContextualChunk],
        task_template: str,
        group_id: str
    ) -> List[ContextualChunk]:
        """
        Process context inheritance for a group of chunks
        
        Args:
            chunks: List of chunks in the group
            task_template: Main task template
            group_id: ID of the context group
            
        Returns:
            List of chunks with context inheritance applied
        """
        if not chunks:
            return chunks
        
        try:
            processed_chunks = []
            current_context = None
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    # First chunk: generate context summary
                    chunk.generates_context = True
                    
                    context_result = await self.generate_context_summary(
                        chunk=chunk,
                        task_template=task_template,
                        previous_context=None
                    )
                    
                    if context_result.success:
                        current_context = context_result.context_summary
                        
                        # Update chunk with context summary
                        chunk.inherited_context = current_context
                        
                        # Add context inheritance metadata
                        chunk.context_inheritance = ContextInheritance(
                            source_chunk_id=chunk.chunk_id,
                            context_chain=[chunk.chunk_id],
                            context_freshness=0,
                            context_relevance_score=context_result.quality_score,
                            generation_timestamp=datetime.now(),
                            context_summary_tokens=context_result.token_count
                        )
                        
                        logger.info(f"Generated context for chunk {chunk.chunk_id}: {context_result.token_count} tokens")
                    else:
                        logger.warning(f"Context generation failed for chunk {chunk.chunk_id}: {context_result.error_message}")
                        current_context = None
                
                else:
                    # Subsequent chunks: inherit context
                    if current_context:
                        chunk.inherited_context = current_context
                        
                        # Add context inheritance metadata
                        chunk.context_inheritance = ContextInheritance(
                            source_chunk_id=chunks[0].chunk_id,
                            context_chain=[chunks[0].chunk_id],
                            context_freshness=i,
                            context_relevance_score=max(0.1, 1.0 - (i * 0.1)),  # Decay relevance
                            generation_timestamp=datetime.now(),
                            context_summary_tokens=self._estimate_tokens(current_context)
                        )
                        
                        logger.debug(f"Inherited context for chunk {chunk.chunk_id}: {i} steps fresh")
                    
                    # Check if context needs refresh
                    refresh_after = self.context_config.get("refresh_after_chunks", 6)
                    if i > 0 and i % refresh_after == 0:
                        logger.info(f"Context refresh needed at chunk {i} in group {group_id}")
                        # Could implement context refresh logic here
                
                processed_chunks.append(chunk)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Context inheritance processing failed for group {group_id}: {e}")
            return chunks