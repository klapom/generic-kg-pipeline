"""
Shared pytest fixtures for all tests
"""
import pytest
import logging
from pathlib import Path
from typing import Generator
import tempfile
import shutil

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vllm.model_manager import VLLMModelManager
from core.config import Config


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent.parent / "data" / "input"


@pytest.fixture
def sample_pdf_path(test_data_dir) -> Path:
    """Path to sample test PDF"""
    pdf_path = test_data_dir / "test_simple.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def bmw_pdf_path(test_data_dir) -> Path:
    """Path to BMW test PDF"""
    pdf_path = test_data_dir / "Preview_BMW_3er_G20.pdf"
    if not pdf_path.exists():
        pytest.skip(f"BMW PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def configured_logger():
    """Configured logger for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("test")


@pytest.fixture
def default_config() -> dict:
    """Default configuration for tests"""
    return {
        'max_pages': 5,
        'gpu_memory_utilization': 0.2,
        'chunk_size': 1000,
        'chunk_overlap': 200
    }


@pytest.fixture
def layout_settings() -> dict:
    """Default layout settings for PDF extraction"""
    return {
        'use_layout': True,
        'table_x_tolerance': 3,
        'table_y_tolerance': 3,
        'text_x_tolerance': 5,
        'text_y_tolerance': 5
    }


@pytest.fixture(scope="session")
def model_manager() -> Generator[VLLMModelManager, None, None]:
    """Shared model manager (session scoped to avoid reloading)"""
    manager = VLLMModelManager()
    yield manager
    # Cleanup will be handled by the manager itself


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM response for testing"""
    return {
        "choices": [{
            "text": "<doctag><text>Sample parsed content</text></doctag>"
        }]
    }


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "requires_model: marks tests that require model download")