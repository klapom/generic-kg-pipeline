#!/usr/bin/env python3
"""
Document Element Classifier for intelligent routing of images to appropriate VLMs.
Classifies images as diagrams, tables, text-heavy, or general content.
"""

import logging
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import io

from core.parsers.interfaces.data_models import VisualElementType

logger = logging.getLogger(__name__)


class DocumentElementClassifier:
    """
    Classifies document images to determine optimal VLM processing strategy.
    Uses heuristics and image analysis for fast classification.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.diagram_keywords = [
            'diagram', 'flow', 'chart', 'graph', 'plot', 'schema',
            'architecture', 'network', 'process', 'workflow', 'tree',
            'timeline', 'mindmap', 'uml', 'circuit', 'blueprint'
        ]
        
        self.table_keywords = [
            'table', 'grid', 'spreadsheet', 'matrix', 'schedule',
            'calendar', 'list', 'inventory', 'specification'
        ]
        
    def detect_element_type(self, image_data: bytes, 
                          context_text: Optional[str] = None) -> VisualElementType:
        """
        Detect the type of visual element in the image.
        
        Args:
            image_data: Raw image bytes
            context_text: Optional surrounding text for context
            
        Returns:
            VisualElementType: Detected element type
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Get image characteristics
            width, height = image.size
            aspect_ratio = width / height
            
            # Convert to numpy for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Analyze image characteristics
            is_diagram = self._check_diagram_characteristics(img_array, aspect_ratio)
            is_table = self._check_table_characteristics(img_array, aspect_ratio)
            is_text_heavy = self._check_text_density(img_array)
            
            # Use context if available
            if context_text:
                context_lower = context_text.lower()
                if any(keyword in context_lower for keyword in self.diagram_keywords):
                    is_diagram = True
                elif any(keyword in context_lower for keyword in self.table_keywords):
                    is_table = True
            
            # Determine type based on analysis
            if is_diagram:
                return VisualElementType.DIAGRAM
            elif is_table:
                return VisualElementType.TABLE
            elif is_text_heavy:
                return VisualElementType.IMAGE  # Text-heavy image
            else:
                return VisualElementType.IMAGE  # General image
                
        except Exception as e:
            logger.warning(f"Error in element type detection: {e}")
            return VisualElementType.IMAGE  # Default to general image
    
    def is_diagram(self, image_data: bytes) -> bool:
        """
        Check if the image is likely a diagram.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            bool: True if likely a diagram
        """
        element_type = self.detect_element_type(image_data)
        return element_type == VisualElementType.DIAGRAM
    
    def requires_high_precision(self, element_type: VisualElementType) -> bool:
        """
        Determine if an element type requires high-precision processing.
        
        Args:
            element_type: Type of visual element
            
        Returns:
            bool: True if high precision (Pixtral) recommended
        """
        high_precision_types = [
            VisualElementType.DIAGRAM,
            VisualElementType.TABLE,
            VisualElementType.CHART
        ]
        return element_type in high_precision_types
    
    def _check_diagram_characteristics(self, img_array: np.ndarray, 
                                     aspect_ratio: float) -> bool:
        """
        Check if image has diagram-like characteristics.
        
        Args:
            img_array: Image as numpy array
            aspect_ratio: Width/height ratio
            
        Returns:
            bool: True if likely a diagram
        """
        try:
            # Check for high contrast (common in diagrams)
            gray = np.mean(img_array, axis=2)
            contrast = np.std(gray)
            
            # Check for geometric shapes (edges)
            edges = self._detect_edges(gray)
            edge_density = np.sum(edges) / edges.size
            
            # Diagrams often have:
            # - High contrast (>50)
            # - Moderate edge density (0.05-0.3)
            # - Often wider than tall (aspect_ratio > 1.2)
            
            is_high_contrast = contrast > 50
            is_moderate_edges = 0.05 < edge_density < 0.3
            is_wide = aspect_ratio > 1.2
            
            # Score based on characteristics
            score = sum([is_high_contrast, is_moderate_edges, is_wide])
            
            return score >= 2
            
        except Exception as e:
            logger.debug(f"Error in diagram detection: {e}")
            return False
    
    def _check_table_characteristics(self, img_array: np.ndarray, 
                                   aspect_ratio: float) -> bool:
        """
        Check if image has table-like characteristics.
        
        Args:
            img_array: Image as numpy array
            aspect_ratio: Width/height ratio
            
        Returns:
            bool: True if likely a table
        """
        try:
            gray = np.mean(img_array, axis=2)
            
            # Detect horizontal and vertical lines
            edges = self._detect_edges(gray)
            horizontal_lines = self._count_lines(edges, axis=0)
            vertical_lines = self._count_lines(edges, axis=1)
            
            # Tables typically have:
            # - Multiple horizontal lines (>3)
            # - Multiple vertical lines (>2)
            # - Regular spacing
            
            has_horizontal = horizontal_lines > 3
            has_vertical = vertical_lines > 2
            
            return has_horizontal and has_vertical
            
        except Exception as e:
            logger.debug(f"Error in table detection: {e}")
            return False
    
    def _check_text_density(self, img_array: np.ndarray) -> bool:
        """
        Check if image is text-heavy.
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            bool: True if text-heavy
        """
        try:
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
            
            # Text areas typically have:
            # - High frequency changes (text edges)
            # - Consistent background
            # - Low color variance
            
            # Calculate local variance
            variance = np.var(gray)
            
            # Text-heavy images have moderate variance (10-60)
            return 10 < variance < 60
            
        except Exception as e:
            logger.debug(f"Error in text density check: {e}")
            return False
    
    def _detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Simple edge detection using gradient.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Edge map
        """
        # Simple Sobel-like edge detection
        dx = np.abs(np.diff(gray_image, axis=1))
        dy = np.abs(np.diff(gray_image, axis=0))
        
        # Combine gradients
        edges = np.zeros_like(gray_image)
        edges[:-1, :-1] = (dx[:-1, :] + dy[:, :-1]) > 30
        
        return edges
    
    def _count_lines(self, edges: np.ndarray, axis: int) -> int:
        """
        Count approximate number of lines along an axis.
        
        Args:
            edges: Edge map
            axis: 0 for horizontal, 1 for vertical
            
        Returns:
            Approximate line count
        """
        # Sum along axis
        projection = np.sum(edges, axis=axis)
        
        # Count peaks (lines)
        threshold = np.max(projection) * 0.3
        lines = projection > threshold
        
        # Count transitions
        transitions = np.sum(np.diff(lines.astype(int)) > 0)
        
        return transitions
    
    def get_processing_recommendation(self, image_data: bytes, 
                                     confidence_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Get comprehensive processing recommendation for an image.
        
        Args:
            image_data: Raw image bytes
            confidence_threshold: Threshold for requiring fallback
            
        Returns:
            Dict with processing recommendations
        """
        element_type = self.detect_element_type(image_data)
        requires_precision = self.requires_high_precision(element_type)
        
        recommendation = {
            "element_type": element_type,
            "primary_model": "pixtral" if requires_precision else "qwen",
            "requires_high_precision": requires_precision,
            "confidence_threshold": confidence_threshold,
            "processing_hints": []
        }
        
        # Add specific hints based on type
        if element_type == VisualElementType.DIAGRAM:
            recommendation["processing_hints"].extend([
                "Focus on relationships and connections",
                "Extract labels and annotations",
                "Identify flow direction"
            ])
        elif element_type == VisualElementType.TABLE:
            recommendation["processing_hints"].extend([
                "Extract structured data",
                "Preserve row/column relationships",
                "Identify headers"
            ])
        elif element_type == VisualElementType.IMAGE:
            recommendation["processing_hints"].extend([
                "General content extraction",
                "Identify key visual elements"
            ])
        
        return recommendation