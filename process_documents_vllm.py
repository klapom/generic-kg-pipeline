#!/usr/bin/env python3
"""
vLLM-Enhanced Document Processing CLI Script

High-performance document processing with local vLLM models:
- SmolDocling for PDF parsing
- Qwen2.5-VL for visual analysis
- Automatic model lifecycle management
- Batch processing optimization

Usage:
    python process_documents_vllm.py --vllm         # Use vLLM models (default)
    python process_documents_vllm.py                # Use vLLM models (default)
    python process_documents_vllm.py --help         # Show all options
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.vllm_batch_processor import (
    VLLMBatchProcessor, 
    BatchProcessingConfig, 
    run_vllm_batch_processing,
    BatchProcessingResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

- **PDF** (.pdf) - Parsed using vLLM SmolDocling (local GPU processing)
- **DOCX** (.docx) - Microsoft Word documents with image extraction
- **XLSX** (.xlsx) - Excel spreadsheets with chart analysis
- **PPTX** (.pptx) - PowerPoint presentations with slide visuals
- **TXT** (.txt) - Plain text files

## Usage

### vLLM Mode (High Performance)
```bash
python process_documents_vllm.py --vllm
python process_documents_vllm.py    # Default mode
```

### Batch Processing
```bash
python process_documents_vllm.py --batch-size 5 --max-concurrent 3
```

## Output

Processed results will be saved to `data/output/` directory with detailed performance metrics.
""")
    
    print(f"📖 Created README: {readme_file}")
    print("✅ Sample documents created successfully!")


def save_results_to_json(results: List[BatchProcessingResult], output_dir: Path, stats: Dict[str, Any]):
    """Save batch processing results to JSON files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    for result in results:
        filename = result.file_path.stem
        
        if result.success:
            output_file = output_dir / f"{filename}_processed.json"
            
            output_data = {
                "file_info": {
                    "filename": result.file_path.name,
                    "file_path": str(result.file_path),
                    "file_size": result.file_path.stat().st_size,
                    "processed_at": datetime.now().isoformat()
                },
                "processing": {
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata
                },
                "document": {
                    "title": result.document.metadata.title if result.document else None,
                    "content_length": len(result.document.content) if result.document else 0,
                    "segments_count": len(result.document.segments) if result.document else 0,
                    "page_count": result.document.metadata.page_count if result.document else 0
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
                    for chunk in (result.chunks or [])
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Saved: {output_file}")
        else:
            # Save error information
            error_file = output_dir / f"{filename}_error.json"
            error_data = {
                "file_info": {
                    "filename": result.file_path.name,
                    "file_path": str(result.file_path),
                    "processed_at": datetime.now().isoformat()
                },
                "error": {
                    "message": result.error_message,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata
                }
            }
            
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
                
            print(f"❌ Error saved: {error_file}")
    
    # Save summary
    summary_file = output_dir / "processing_summary.json"
    summary_data = {
        "batch_summary": {
            "total_files": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "processing_time": datetime.now().isoformat()
        },
        "performance_stats": stats,
        "files": [
            {
                "filename": r.file_path.name,
                "success": r.success,
                "processing_time": r.processing_time,
                "error": r.error_message if not r.success else None
            }
            for r in results
        ]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved: {summary_file}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="vLLM-Enhanced Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --vllm                    # Use vLLM models (default)
  %(prog)s                           # Use vLLM models (default)
  %(prog)s --max-concurrent 5        # Process with 5 concurrent files
  %(prog)s --create-samples          # Create sample documents
  %(prog)s --file sample.pdf         # Process specific file
        """
    )
    
    # Processing modes
    parser.add_argument("--vllm", action="store_true", default=True, help="Use vLLM models (requires GPU)")
    
    # File options
    parser.add_argument("--file", type=str, help="Process specific file instead of all files")
    parser.add_argument("--create-samples", action="store_true", help="Create sample documents")
    
    # Performance options
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent processing")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization (0.1-1.0)")
    
    # Feature options
    parser.add_argument("--no-chunking", action="store_true", help="Disable content chunking")
    parser.add_argument("--no-context-inheritance", action="store_true", help="Disable context inheritance")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="data/output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create samples if requested
    if args.create_samples:
        create_sample_documents()
        return
    
    print("🚀 vLLM-Enhanced Document Processing")
    print("=" * 50)
    
    # Create processing configuration
    config = BatchProcessingConfig(
        use_vllm=args.vllm,
        max_concurrent=args.max_concurrent,
        enable_chunking=not args.no_chunking,
        enable_context_inheritance=not args.no_context_inheritance,
        gpu_memory_utilization=args.gpu_memory
    )
    
    # Print configuration
    print(f"Mode: vLLM")
    print(f"Chunking: {'Enabled' if config.enable_chunking else 'Disabled'}")
    print(f"Context Inheritance: {'Enabled' if config.enable_context_inheritance else 'Disabled'}")
    print(f"Max Concurrent: {config.max_concurrent}")
    print(f"GPU Memory: {config.gpu_memory_utilization:.1%}")
    print()
    
    input_dir = Path("data/input")
    output_dir = Path(args.output_dir)
    
    try:
        if args.file:
            # Process specific file
            file_path = input_dir / args.file
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return
            
            print(f"Processing single file: {file_path.name}")
            
            # Create processor
            processor = VLLMBatchProcessor(config)
            results = await processor.run_complete_batch([file_path])
            
        else:
            # Process all files using the convenience function
            print("Processing all files in batch mode...")
            results = await run_vllm_batch_processing(input_dir, output_dir, config)
        
        # Save results
        if results:
            # Get statistics from processor if available
            stats = {
                "total_files": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "total_processing_time": sum(r.processing_time for r in results),
                "average_processing_time": sum(r.processing_time for r in results) / len(results)
            }
            
            save_results_to_json(results, output_dir, stats)
            
            # Print final summary
            print("\n📊 Final Summary:")
            print(f"   Total files: {stats['total_files']}")
            print(f"   Successful: {stats['successful']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Success rate: {(stats['successful']/stats['total_files']*100):.1f}%")
            print(f"   Total time: {stats['total_processing_time']:.1f}s")
            print(f"   Average per file: {stats['average_processing_time']:.1f}s")
            print(f"   Results saved to: {output_dir}")
        else:
            print("⚠️  No files processed")
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())