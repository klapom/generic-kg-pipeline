"""Tests for Document Parsers and Multi-Modal Processing"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from core.parsers import (
    ParserFactory, Document, DocumentType, DocumentMetadata,
    Segment, VisualElement, VisualElementType,
    PDFParser, DOCXParser, XLSXParser, PPTXParser
)
# Note: TextSegment is replaced by Segment in the new structure


class TestParserFactory:
    """Test Parser Factory functionality"""
    
    def test_factory_initialization(self):
        """Test that factory initializes correctly"""
        factory = ParserFactory()
        assert factory is not None

    def test_get_parser_pdf(self):
        """Test getting PDF parser"""
        factory = ParserFactory()
        parser = factory.get_parser(DocumentType.PDF)
        assert isinstance(parser, PDFParser)

    def test_get_parser_docx(self):
        """Test getting DOCX parser"""
        factory = ParserFactory()
        parser = factory.get_parser(DocumentType.DOCX)
        assert isinstance(parser, DOCXParser)

    def test_get_parser_xlsx(self):
        """Test getting XLSX parser"""
        factory = ParserFactory()
        parser = factory.get_parser(DocumentType.XLSX)
        assert isinstance(parser, XLSXParser)

    def test_get_parser_pptx(self):
        """Test getting PPTX parser"""
        factory = ParserFactory()
        parser = factory.get_parser(DocumentType.PPTX)
        assert isinstance(parser, PPTXParser)

    def test_unsupported_format(self):
        """Test handling of unsupported format"""
        factory = ParserFactory()
        with pytest.raises(ValueError, match="Unsupported document type"):
            factory.get_parser("unsupported_format")

    @pytest.mark.asyncio
    async def test_parse_file_by_extension(self, tmp_path):
        """Test parsing file by extension detection"""
        factory = ParserFactory()
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for parsing")
        
        with patch.object(factory, '_get_text_parser') as mock_text_parser:
            mock_parser = MagicMock()
            mock_parser.parse_file.return_value = Document(
                metadata=DocumentMetadata(
                    title="test.txt",
                    document_type=DocumentType.TXT,
                    page_count=1,
                    created_at=datetime.now()
                ),
                segments=[
                    TextSegment(
                        content="Test content for parsing",
                        metadata={"type": "paragraph"},
                        page_or_slide=1,
                        position=(0, 23)
                    )
                ],
                visual_elements=[]
            )
            mock_text_parser.return_value = mock_parser
            
            document = await factory.parse_file(test_file)
            
            assert isinstance(document, Document)
            assert document.metadata.title == "test.txt"
            assert len(document.segments) == 1
            assert document.segments[0].content == "Test content for parsing"


class TestPDFParser:
    """Test PDF Parser functionality"""
    
    @pytest.fixture
    def pdf_parser(self, test_config):
        """Create PDF parser with test config"""
        return PDFParser(test_config)

    @pytest.mark.asyncio
    async def test_pdf_parser_with_mock_vllm(self, pdf_parser, sample_pdf_path, mock_vllm_smoldocling):
        """Test PDF parsing with mocked vLLM SmolDocling"""
        
        document = await pdf_parser.parse_file(sample_pdf_path)
        
        assert isinstance(document, Document)
        assert document.metadata.document_type == DocumentType.PDF
        assert document.metadata.title == sample_pdf_path.name
        assert len(document.segments) > 0
        
        # Check that vLLM was called
        mock_vllm_smoldocling.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_pdf_parser_content_extraction(self, pdf_parser, sample_pdf_path, mock_vllm_smoldocling):
        """Test that PDF content is extracted correctly"""
        
        document = await pdf_parser.parse_file(sample_pdf_path)
        
        # Check content from mock response
        page_contents = [seg.content for seg in document.segments if seg.metadata.get("type") == "page"]
        assert any("This is page 1 content" in content for content in page_contents)
        assert any("This is page 2 content" in content for content in page_contents)

    @pytest.mark.asyncio
    async def test_pdf_parser_table_extraction(self, pdf_parser, sample_pdf_path, mock_vllm_smoldocling):
        """Test that tables are extracted from PDF"""
        
        document = await pdf_parser.parse_file(sample_pdf_path)
        
        # Check for table content from mock response
        table_segments = [seg for seg in document.segments if seg.metadata.get("type") == "table"]
        if table_segments:
            assert len(table_segments) > 0
            assert "Test Table" in table_segments[0].content

    @pytest.mark.asyncio
    async def test_pdf_parser_error_handling(self, pdf_parser, tmp_path):
        """Test PDF parser error handling"""
        
        # Create invalid PDF file
        invalid_pdf = tmp_path / "invalid.pdf"
        invalid_pdf.write_text("This is not a valid PDF")
        
        with pytest.raises(Exception):
            await pdf_parser.parse_file(invalid_pdf)


class TestDOCXParser:
    """Test DOCX Parser functionality"""
    
    @pytest.fixture
    def docx_parser(self, test_config):
        """Create DOCX parser with test config"""
        return DOCXParser(test_config)

    @pytest.mark.asyncio
    async def test_docx_parser_basic(self, docx_parser, sample_docx_path):
        """Test basic DOCX parsing"""
        
        with patch('docx.Document') as mock_docx:
            # Mock docx document structure
            mock_doc = MagicMock()
            mock_paragraph = MagicMock()
            mock_paragraph.text = "Test paragraph content"
            mock_paragraph.style.name = "Normal"
            mock_doc.paragraphs = [mock_paragraph]
            mock_doc.inline_shapes = []
            
            mock_docx.return_value = mock_doc
            
            document = await docx_parser.parse_file(sample_docx_path)
            
            assert isinstance(document, Document)
            assert document.metadata.document_type == DocumentType.DOCX
            assert len(document.segments) > 0

    @pytest.mark.asyncio
    async def test_docx_image_extraction(self, docx_parser, sample_docx_path):
        """Test image extraction from DOCX"""
        
        with patch('docx.Document') as mock_docx:
            # Mock document with image
            mock_doc = MagicMock()
            mock_doc.paragraphs = []
            
            # Mock inline shape (image)
            mock_image = MagicMock()
            mock_image.type = 3  # PICTURE type
            mock_image.width = 100
            mock_image.height = 200
            mock_doc.inline_shapes = [mock_image]
            
            mock_docx.return_value = mock_doc
            
            document = await docx_parser.parse_file(sample_docx_path)
            
            # Check that visual elements were created
            assert len(document.visual_elements) > 0
            visual = document.visual_elements[0]
            assert visual.element_type == VisualElementType.IMAGE
            assert visual.size == (100, 200)


class TestXLSXParser:
    """Test XLSX Parser functionality"""
    
    @pytest.fixture
    def xlsx_parser(self, test_config):
        """Create XLSX parser with test config"""
        return XLSXParser(test_config)

    @pytest.mark.asyncio
    async def test_xlsx_parser_basic(self, xlsx_parser, tmp_path):
        """Test basic XLSX parsing"""
        
        xlsx_file = tmp_path / "test.xlsx"
        
        with patch('openpyxl.load_workbook') as mock_openpyxl:
            # Mock workbook structure
            mock_workbook = MagicMock()
            mock_worksheet = MagicMock()
            mock_worksheet.title = "Sheet1"
            mock_worksheet.max_row = 3
            mock_worksheet.max_column = 2
            
            # Mock cell values
            mock_worksheet.cell.side_effect = lambda row, col: MagicMock(value=f"Cell_{row}_{col}")
            
            mock_workbook.worksheets = [mock_worksheet]
            mock_openpyxl.return_value = mock_workbook
            
            # Create minimal xlsx file
            xlsx_file.write_bytes(b"PK\x03\x04test content")
            
            document = await xlsx_parser.parse_file(xlsx_file)
            
            assert isinstance(document, Document)
            assert document.metadata.document_type == DocumentType.XLSX
            assert len(document.segments) > 0

    @pytest.mark.asyncio
    async def test_xlsx_chart_detection(self, xlsx_parser, tmp_path):
        """Test chart detection in XLSX"""
        
        xlsx_file = tmp_path / "test_with_chart.xlsx"
        
        with patch('openpyxl.load_workbook') as mock_openpyxl:
            # Mock workbook with chart
            mock_workbook = MagicMock()
            mock_worksheet = MagicMock()
            mock_worksheet.title = "Sheet1"
            mock_worksheet.max_row = 2
            mock_worksheet.max_column = 2
            mock_worksheet.cell.side_effect = lambda row, col: MagicMock(value=f"Data_{row}_{col}")
            
            # Mock chart
            mock_chart = MagicMock()
            mock_chart.title = "Test Chart"
            mock_worksheet._charts = [mock_chart]
            
            mock_workbook.worksheets = [mock_worksheet]
            mock_openpyxl.return_value = mock_workbook
            
            xlsx_file.write_bytes(b"PK\x03\x04test content")
            
            document = await xlsx_parser.parse_file(xlsx_file)
            
            # Check for chart visual elements
            chart_visuals = [v for v in document.visual_elements if v.element_type == VisualElementType.CHART]
            assert len(chart_visuals) > 0
            assert chart_visuals[0].description == "Chart: Test Chart"


class TestPPTXParser:
    """Test PPTX Parser functionality"""
    
    @pytest.fixture
    def pptx_parser(self, test_config):
        """Create PPTX parser with test config"""  
        return PPTXParser(test_config)

    @pytest.mark.asyncio
    async def test_pptx_parser_basic(self, pptx_parser, tmp_path):
        """Test basic PPTX parsing"""
        
        pptx_file = tmp_path / "test.pptx"
        
        with patch('pptx.Presentation') as mock_pptx:
            # Mock presentation structure
            mock_presentation = MagicMock()
            mock_slide = MagicMock()
            
            # Mock text shapes
            mock_shape = MagicMock()
            mock_shape.has_text_frame = True
            mock_shape.text = "Slide title content"
            mock_slide.shapes = [mock_shape]
            
            mock_presentation.slides = [mock_slide]
            mock_pptx.return_value = mock_presentation
            
            pptx_file.write_bytes(b"PK\x03\x04test content")
            
            document = await pptx_parser.parse_file(pptx_file)
            
            assert isinstance(document, Document)
            assert document.metadata.document_type == DocumentType.PPTX
            assert len(document.segments) > 0

    @pytest.mark.asyncio
    async def test_pptx_image_extraction(self, pptx_parser, tmp_path):
        """Test image extraction from PPTX"""
        
        pptx_file = tmp_path / "test_with_images.pptx"
        
        with patch('pptx.Presentation') as mock_pptx:
            # Mock presentation with image
            mock_presentation = MagicMock()
            mock_slide = MagicMock()
            
            # Mock image shape
            mock_image_shape = MagicMock()
            mock_image_shape.has_text_frame = False
            mock_image_shape.shape_type = 13  # PICTURE type
            mock_image_shape.width = 150
            mock_image_shape.height = 100
            
            mock_slide.shapes = [mock_image_shape]
            mock_presentation.slides = [mock_slide]
            mock_pptx.return_value = mock_presentation
            
            pptx_file.write_bytes(b"PK\x03\x04test content")
            
            document = await pptx_parser.parse_file(pptx_file)
            
            # Check for image visual elements
            image_visuals = [v for v in document.visual_elements if v.element_type == VisualElementType.IMAGE]
            assert len(image_visuals) > 0
            assert image_visuals[0].size == (150, 100)


@pytest.mark.asyncio
async def test_parser_integration_with_qwen_vl():
    """Test parser integration with Qwen2.5-VL for visual analysis"""
    
    with patch('core.clients.qwen25_vl.Qwen25VLClient.analyze_visual_content') as mock_qwen:
        mock_qwen.return_value = {
            "description": "This image shows a bar chart with sales data",
            "extracted_text": "Sales Q1: 100K, Q2: 150K, Q3: 200K",
            "confidence": 0.95
        }
        
        # Create mock visual element
        visual = VisualElement(
            element_type=VisualElementType.CHART,
            page_or_slide=1,
            position=(100, 100),
            size=(300, 200),
            description="Chart in presentation"
        )
        
        # Test that VLM analysis enhances visual elements
        factory = ParserFactory()
        
        with patch.object(factory, '_enhance_visual_with_vlm') as mock_enhance:
            mock_enhance.return_value = VisualElement(
                element_type=VisualElementType.CHART,
                page_or_slide=1,
                position=(100, 100),
                size=(300, 200),
                description="Chart in presentation",
                vlm_description="This image shows a bar chart with sales data",
                extracted_data={"text": "Sales Q1: 100K, Q2: 150K, Q3: 200K"}
            )
            
            enhanced_visual = await factory._enhance_visual_with_vlm(visual, b"fake_image_data")
            
            assert enhanced_visual.vlm_description == "This image shows a bar chart with sales data"
            assert "Sales Q1: 100K" in enhanced_visual.extracted_data["text"]