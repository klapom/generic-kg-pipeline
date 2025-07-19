"""Parser factory for multi-modal document processing"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .interfaces.base_parser import BaseParser
from .interfaces.data_models import DocumentType, ParseError

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory class for creating document parsers
    
    Automatically selects the appropriate parser based on file type
    and provides unified interface for multi-modal document processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True):
        """
        Initialize parser factory
        
        Args:
            config: Configuration dictionary for all parsers
            enable_vlm: Whether to enable VLM analysis for visual elements
        """
        self.config = config or {}
        self.enable_vlm = enable_vlm
        
        # Register available parsers - all loaded dynamically
        self._parsers: Dict[DocumentType, Type[BaseParser]] = {}
        
        # Load parsers dynamically to avoid circular imports
        try:
            # Check if we should use the new PDF processor architecture
            use_new_pdf_processor = self.config.get('use_new_pdf_processor', True)
            use_image_extraction = self.config.get('use_image_extraction_parser', False)
            
            if use_new_pdf_processor:
                from .implementations.pdf import PDFProcessor
                self._parsers[DocumentType.PDF] = PDFProcessor
                logger.info("Using new PDFProcessor architecture")
            elif use_image_extraction:
                # Legacy option for backward compatibility
                from .implementations.pdf import ImageExtractionPDFParser
                self._parsers[DocumentType.PDF] = ImageExtractionPDFParser
                logger.info("Using ImageExtractionPDFParser for visual element extraction")
            else:
                # Legacy option for backward compatibility
                from .implementations.pdf import HybridPDFParser
                self._parsers[DocumentType.PDF] = HybridPDFParser
                logger.info("Using standard HybridPDFParser")
        except ImportError as e:
            logger.warning(f"PDF Parser not available: {e}")
            
        try:
            from .implementations.text import TXTParser
            self._parsers[DocumentType.TXT] = TXTParser
        except ImportError:
            logger.warning("TXTParser not available")
            
        try:
            from .implementations.office import DOCXParser
            self._parsers[DocumentType.DOCX] = DOCXParser
        except ImportError:
            logger.warning("DOCXParser not available")
            
        try:
            from .implementations.office import XLSXParser
            self._parsers[DocumentType.XLSX] = XLSXParser
        except ImportError:
            logger.warning("XLSXParser not available")
            
        try:
            from .implementations.office import PPTXParser
            self._parsers[DocumentType.PPTX] = PPTXParser
        except ImportError:
            logger.warning("PPTXParser not available")
        
        # Parser instances cache
        self._parser_instances: Dict[DocumentType, BaseParser] = {}
        
        logger.info(f"Initialized parser factory with {len(self._parsers)} parsers, VLM: {enable_vlm}")
    
    def get_supported_types(self) -> List[DocumentType]:
        """Get list of supported document types"""
        return list(self._parsers.keys())
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        extensions = []
        for doc_type in self._parsers.keys():
            extensions.append(f".{doc_type.value}")
        return extensions
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file can be parsed by any available parser
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file can be parsed, False otherwise
        """
        try:
            parser = self._get_parser_for_file(file_path)
            return parser is not None
        except Exception:
            return False
    
    def get_parser_for_file(self, file_path: Path) -> BaseParser:
        """
        Get appropriate parser for the given file
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Parser instance for the file
            
        Raises:
            ParseError: If no suitable parser found
        """
        parser = self._get_parser_for_file(file_path)
        if parser is None:
            raise ParseError(f"No parser available for file: {file_path.name}")
        return parser
    
    def _get_parser_for_file(self, file_path: Path) -> Optional[BaseParser]:
        """Internal method to get parser for file"""
        try:
            # Determine document type from extension
            doc_type = self._get_document_type(file_path)
            if doc_type == DocumentType.UNKNOWN:
                return None
            
            # Get or create parser instance
            return self._get_parser_instance(doc_type)
            
        except Exception as e:
            logger.error(f"Failed to get parser for {file_path}: {e}")
            return None
    
    def _get_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension"""
        extension = file_path.suffix.lower().lstrip('.')
        
        # Map extensions to document types
        extension_map = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.DOCX,
            'doc': DocumentType.DOCX,  # Treat .doc as DOCX
            'xlsx': DocumentType.XLSX,
            'xls': DocumentType.XLSX,  # Treat .xls as XLSX
            'pptx': DocumentType.PPTX,
            'ppt': DocumentType.PPTX,  # Treat .ppt as PPTX
            'txt': DocumentType.TXT
        }
        
        return extension_map.get(extension, DocumentType.UNKNOWN)
    
    def _get_parser_instance(self, doc_type: DocumentType) -> Optional[BaseParser]:
        """Get or create parser instance for document type"""
        try:
            # Check if we have a parser for this type
            if doc_type not in self._parsers:
                return None
            
            # Return cached instance if available
            if doc_type in self._parser_instances:
                return self._parser_instances[doc_type]
            
            # Create new parser instance
            parser_class = self._parsers[doc_type]
            parser_config = self.config.get(doc_type.value, {})
            
            parser_instance = parser_class(
                config=parser_config,
                enable_vlm=self.enable_vlm
            )
            
            # Cache the instance
            self._parser_instances[doc_type] = parser_instance
            
            return parser_instance
            
        except Exception as e:
            logger.error(f"Failed to create parser for {doc_type}: {e}")
            return None
    
    async def parse_document(self, file_path: Path) -> Any:
        """
        Parse document using appropriate parser
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Parsed Document object
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            # Get appropriate parser
            parser = self.get_parser_for_file(file_path)
            
            logger.info(f"Parsing {file_path.name} with {type(parser).__name__}")
            
            # Parse the document
            # Check if parser has async parse method
            if hasattr(parser, 'parse_async'):
                document = await parser.parse_async(file_path)
            else:
                # Fall back to sync parse for legacy parsers
                document = parser.parse(file_path)
            
            logger.info(f"Successfully parsed {file_path.name}: "
                       f"{document.total_segments} segments, "
                       f"{document.total_visual_elements} visual elements")
            
            return document
            
        except Exception as e:
            logger.error(f"Document parsing failed for {file_path}: {e}")
            raise ParseError(f"Failed to parse {file_path.name}: {str(e)}")
    
    async def parse_multiple_documents(self, file_paths: List[Path]) -> List[Any]:
        """
        Parse multiple documents concurrently
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of parsed Document objects
            
        Raises:
            ParseError: If any parsing fails
        """
        import asyncio
        
        try:
            logger.info(f"Parsing {len(file_paths)} documents")
            
            # Create parsing tasks
            tasks = []
            for file_path in file_paths:
                task = self.parse_document(file_path)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            documents = []
            failed_files = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_files.append((file_paths[i], result))
                    logger.error(f"Failed to parse {file_paths[i].name}: {result}")
                else:
                    documents.append(result)
            
            if failed_files:
                logger.warning(f"Failed to parse {len(failed_files)} out of {len(file_paths)} documents")
                for file_path, error in failed_files:
                    logger.warning(f"  - {file_path.name}: {error}")
            
            logger.info(f"Successfully parsed {len(documents)} out of {len(file_paths)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Batch parsing failed: {e}")
            raise ParseError(f"Failed to parse multiple documents: {str(e)}")
    
    def get_parser_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about the parser that would be used for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with parser information
        """
        try:
            doc_type = self._get_document_type(file_path)
            
            if doc_type == DocumentType.UNKNOWN:
                return {
                    "file_path": str(file_path),
                    "document_type": "unknown",
                    "parser_available": False,
                    "supported": False
                }
            
            parser_class = self._parsers.get(doc_type)
            
            return {
                "file_path": str(file_path),
                "document_type": doc_type.value,
                "parser_class": parser_class.__name__ if parser_class else None,
                "parser_available": parser_class is not None,
                "supported": parser_class is not None,
                "vlm_enabled": self.enable_vlm,
                "config": self.config.get(doc_type.value, {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get parser info for {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "supported": False
            }
    
    def get_factory_status(self) -> Dict[str, Any]:
        """
        Get status information about the factory
        
        Returns:
            Dictionary with factory status
        """
        return {
            "supported_types": [dt.value for dt in self.get_supported_types()],
            "supported_extensions": self.get_supported_extensions(),
            "vlm_enabled": self.enable_vlm,
            "parser_count": len(self._parsers),
            "cached_instances": len(self._parser_instances),
            "parsers": {
                doc_type.value: parser_class.__name__
                for doc_type, parser_class in self._parsers.items()
            }
        }
    
    def clear_cache(self):
        """Clear cached parser instances"""
        self._parser_instances.clear()
        logger.info("Parser instance cache cleared")
    
    def register_parser(self, doc_type: DocumentType, parser_class: Type[BaseParser]):
        """
        Register a new parser for a document type
        
        Args:
            doc_type: Document type to register parser for
            parser_class: Parser class to register
        """
        self._parsers[doc_type] = parser_class
        # Clear cached instance if exists
        if doc_type in self._parser_instances:
            del self._parser_instances[doc_type]
        
        logger.info(f"Registered parser {parser_class.__name__} for {doc_type.value}")
    
    def unregister_parser(self, doc_type: DocumentType):
        """
        Unregister parser for a document type
        
        Args:
            doc_type: Document type to unregister parser for
        """
        if doc_type in self._parsers:
            del self._parsers[doc_type]
        
        if doc_type in self._parser_instances:
            del self._parser_instances[doc_type]
        
        logger.info(f"Unregistered parser for {doc_type.value}")


# Global factory instance
_default_factory: Optional[ParserFactory] = None


def get_default_factory(config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True) -> ParserFactory:
    """
    Get the default parser factory instance
    
    Args:
        config: Configuration dictionary for parsers
        enable_vlm: Whether to enable VLM analysis
        
    Returns:
        Default ParserFactory instance
    """
    global _default_factory
    
    if _default_factory is None:
        _default_factory = ParserFactory(config=config, enable_vlm=enable_vlm)
    
    return _default_factory


def set_default_factory(factory: ParserFactory):
    """
    Set the default parser factory instance
    
    Args:
        factory: ParserFactory instance to set as default
    """
    global _default_factory
    _default_factory = factory


def parse_document(file_path: Path, config: Optional[Dict[str, Any]] = None, enable_vlm: bool = True) -> Any:
    """
    Convenience function to parse a document using the default factory
    
    Args:
        file_path: Path to the document file
        config: Configuration dictionary for parsers
        enable_vlm: Whether to enable VLM analysis
        
    Returns:
        Parsed Document object
    """
    factory = get_default_factory(config=config, enable_vlm=enable_vlm)
    return factory.parse_document(file_path)


def can_parse(file_path: Path) -> bool:
    """
    Convenience function to check if a file can be parsed
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file can be parsed, False otherwise
    """
    factory = get_default_factory()
    return factory.can_parse(file_path)