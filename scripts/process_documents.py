#!/usr/bin/env python3
"""
Document Processing CLI Script
Processes documents from data/input/ directory and saves results to data/output/

Now with vLLM integration for high-performance local model processing!
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config_new.unified_manager import get_config
from core.content_chunker import ContentChunker
from core.vllm_batch_processor import VLLMBatchProcessor, BatchProcessingConfig, run_vllm_batch_processing
from plugins.parsers.parser_factory import ParserFactory
from plugins.parsers.base_parser import Document


class DocumentProcessor:
    """Command line document processor"""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = get_config()
        self.config_override = config_override or {}
        self.parser_factory = ParserFactory()
        self.input_dir = Path("data/input")
        self.output_dir = Path("data/output")
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_supported_files(self) -> List[Path]:
        """Get all supported document files from input directory"""
        supported_extensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt']
        files = []
        
        for ext in supported_extensions:
            files.extend(self.input_dir.glob(f"*{ext}"))
        
        return sorted(files)
    
    async def process_file(self, file_path: Path, mock_mode: bool = False) -> Dict[str, Any]:
        """Process a single document file"""
        print(f"🔄 Processing: {file_path.name}")
        
        try:
            # Parse the document
            parser = self.parser_factory.get_parser_for_file(file_path)
            
            if mock_mode:
                # Create mock document for testing without external services
                from plugins.parsers.base_parser import DocumentMetadata, DocumentType, Segment
                
                metadata = DocumentMetadata(
                    title=file_path.stem,
                    document_type=DocumentType.from_extension(file_path.suffix),
                    page_count=1,
                    file_path=file_path,
                    created_date=datetime.now()
                )
                
                # Create simple mock content
                mock_content = f"Mock content for {file_path.name}. This is a test document."
                segments = [
                    Segment(
                        content=mock_content,
                        page_number=1,
                        segment_index=0,
                        segment_type="text"
                    )
                ]
                
                document = Document(
                    content=mock_content,
                    metadata=metadata,
                    segments=segments
                )
                
                parsing_result = {
                    "status": "success",
                    "mock_mode": True,
                    "segments_count": len(segments),
                    "content_length": len(mock_content)
                }
                
            else:
                # Real parsing (requires external services)
                document = await parser.parse_document(file_path)
                parsing_result = {
                    "status": "success",
                    "mock_mode": False,
                    "segments_count": len(document.segments),
                    "content_length": len(document.content),
                    "metadata": {
                        "title": document.metadata.title,
                        "page_count": document.metadata.page_count,
                        "document_type": document.metadata.document_type.value
                    }
                }
            
            # Create chunking configuration
            chunking_config = {
                "chunking": {
                    "strategies": {
                        file_path.suffix[1:]: {  # Remove the dot from extension
                            "max_tokens": 500,
                            "min_tokens": 100,
                            "overlap_tokens": 50,
                            "respect_boundaries": True
                        }
                    },
                    "enable_context_inheritance": not mock_mode,  # Disable for mock mode
                    "context_inheritance": {
                        "enabled": not mock_mode,
                        "max_context_tokens": 300,
                        "llm": {
                            "model": "hochschul-llm",
                            "temperature": 0.1
                        }
                    },
                    "performance": {
                        "enable_async_processing": True,
                        "max_concurrent_groups": 3
                    }
                }
            }
            
            # Apply config override
            if self.config_override:
                chunking_config.update(self.config_override)
            
            # Chunk the document
            chunker = ContentChunker(chunking_config)
            chunking_result = await chunker.chunk_document(document)
            
            # Prepare output data
            output_data = {
                "file_info": {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "processed_at": datetime.now().isoformat()
                },
                "parsing": parsing_result,
                "chunking": {
                    "document_id": chunking_result.document_id,
                    "chunks_count": len(chunking_result.contextual_chunks),
                    "context_groups_count": len(chunking_result.context_groups),
                    "processing_stats": {
                        "total_processing_time": chunking_result.processing_stats.total_processing_time,
                        "context_inheritance_time": chunking_result.processing_stats.context_inheritance_time,
                        "grouping_time": chunking_result.processing_stats.grouping_time
                    }
                },
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "chunk_type": chunk.chunk_type.value,
                        "inherited_context": chunk.inherited_context,
                        "generates_context": chunk.generates_context
                    }
                    for chunk in chunking_result.contextual_chunks
                ],
                "context_groups": [
                    {
                        "group_id": group.group_id,
                        "group_type": group.group_type.value,
                        "chunks_count": len(group.chunks),
                        "context_summary": group.context_summary
                    }
                    for group in chunking_result.context_groups
                ]
            }
            
            # Save results
            output_file = self.output_dir / f"{file_path.stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Processed: {file_path.name}")
            print(f"   - Chunks: {len(chunking_result.contextual_chunks)}")
            print(f"   - Groups: {len(chunking_result.context_groups)}")
            print(f"   - Output: {output_file}")
            
            return output_data
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            print(f"❌ {error_msg}")
            
            if "--verbose" in sys.argv:
                print(f"   Traceback: {traceback.format_exc()}")
            
            # Save error information
            error_file = self.output_dir / f"{file_path.stem}_error.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "file_info": {
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "processed_at": datetime.now().isoformat()
                    },
                    "error": {
                        "message": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }
                }, f, indent=2, ensure_ascii=False)
            
            return {
                "file_info": {"filename": file_path.name},
                "error": error_msg
            }
    
    async def process_all_files(self, mock_mode: bool = False) -> Dict[str, Any]:
        """Process all files in the input directory"""
        files = self.get_supported_files()
        
        if not files:
            print("⚠️  No supported files found in data/input/")
            print("   Supported formats: .pdf, .docx, .xlsx, .pptx, .txt")
            return {"processed": 0, "errors": 0, "files": []}
        
        print(f"📁 Found {len(files)} files to process")
        
        if mock_mode:
            print("🎭 Running in MOCK MODE (no external services required)")
        
        results = []
        errors = 0
        
        for file_path in files:
            result = await self.process_file(file_path, mock_mode)
            results.append(result)
            
            if "error" in result:
                errors += 1
        
        summary = {
            "processed": len(files),
            "successful": len(files) - errors,
            "errors": errors,
            "files": results,
            "processing_time": datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total files: {len(files)}")
        print(f"   Successful: {len(files) - errors}")
        print(f"   Errors: {errors}")
        print(f"   Results saved to: {self.output_dir}")
        
        return summary


def create_sample_documents():
    """Create sample documents for testing"""
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample text file
    sample_txt = input_dir / "sample_document.txt"
    with open(sample_txt, 'w', encoding='utf-8') as f:
        f.write("""Machine Learning Overview

Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: Classification, Regression
   - Algorithms: Linear Regression, Decision Trees, Neural Networks

2. Unsupervised Learning
   - Works with unlabeled data
   - Examples: Clustering, Dimensionality Reduction
   - Algorithms: K-means, PCA, Autoencoders

3. Reinforcement Learning
   - Learning through interaction with environment
   - Uses rewards and penalties
   - Examples: Game playing, Robotics

Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep networks) to progressively extract higher-level features from raw input.

Key advantages:
- Automatic feature extraction
- Scalability with large datasets
- State-of-the-art performance in many domains

Applications:
- Computer Vision
- Natural Language Processing
- Speech Recognition
- Autonomous Vehicles

Conclusion

Machine learning continues to evolve rapidly, with new architectures and techniques being developed regularly. Understanding these fundamentals provides a solid foundation for exploring more advanced topics.
""")
    
    print(f"📄 Created sample document: {sample_txt}")
    
    # Create README for input directory
    readme_file = input_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("""# Input Documents Directory

Place your test documents here for processing.

## Supported Formats

- **PDF** (.pdf) - Parsed using vLLM SmolDocling (requires external service)
- **DOCX** (.docx) - Microsoft Word documents with image extraction
- **XLSX** (.xlsx) - Excel spreadsheets with chart analysis
- **PPTX** (.pptx) - PowerPoint presentations with slide visuals
- **TXT** (.txt) - Plain text files

## Usage

1. Copy your documents to this directory
2. Run the processing script:
   ```bash
   python process_documents.py
   ```

## Mock Mode

For testing without external services:
```bash
python process_documents.py --mock
```

## Output

Processed results will be saved to `data/output/` directory.
""")
    
    print(f"📖 Created README: {readme_file}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Process documents from data/input/ directory")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no external services)")
    parser.add_argument("--file", type=str, help="Process specific file instead of all files")
    parser.add_argument("--create-samples", action="store_true", help="Create sample documents")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error messages")
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_documents()
        return
    
    print("🚀 Generic Knowledge Graph Pipeline - Document Processor")
    print("=" * 60)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    try:
        if args.file:
            # Process specific file
            file_path = Path("data/input") / args.file
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return
            
            result = await processor.process_file(file_path, args.mock)
            
            if "error" not in result:
                print(f"\n✅ Successfully processed: {args.file}")
            else:
                print(f"\n❌ Failed to process: {args.file}")
        else:
            # Process all files
            await processor.process_all_files(args.mock)
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())