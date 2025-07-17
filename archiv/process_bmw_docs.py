#!/usr/bin/env python3
"""
Process all Preview_BMW*.pdf files from data/input/
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

from core.parsers import ParserFactory
from core.vlm.batch_processor import BatchDocumentProcessor, BatchProcessingConfig
from core.content_chunker import ContentChunker
from core.vllm_batch_processor import VLLMBatchProcessor, BatchProcessingConfig as VLLMConfig

async def process_bmw_documents():
    """Process all BMW preview documents"""
    
    # Setup paths
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all Preview_BMW*.pdf files
    bmw_files = list(input_dir.glob("Preview_BMW*.pdf"))
    
    if not bmw_files:
        print("❌ No Preview_BMW*.pdf files found in data/input/")
        return
    
    print(f"Found {len(bmw_files)} BMW documents:")
    for f in bmw_files:
        print(f"  - {f.name}")
    
    # Option 1: Basic parsing without VLM
    print("\n1️⃣ Basic Document Parsing...")
    factory = ParserFactory(enable_vlm=False)
    
    for pdf_file in bmw_files:
        print(f"\nProcessing {pdf_file.name}...")
        try:
            document = await factory.parse_document(pdf_file)
            
            # Save parsed document
            output_file = output_dir / f"{pdf_file.stem}_parsed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "file": pdf_file.name,
                    "metadata": document.metadata.to_dict(),
                    "total_segments": len(document.segments),
                    "total_visual_elements": len(document.visual_elements),
                    "segments": [s.to_dict() for s in document.segments[:10]],  # First 10 segments
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
    
    # Option 2: With VLM analysis
    choice = input("\nDo you want to run VLM analysis? (y/n): ")
    if choice.lower() == 'y':
        print("\n2️⃣ Running VLM Analysis...")
        
        config = BatchProcessingConfig(
            batch_size=5,
            use_two_stage_processing=True,
            confidence_threshold=0.7,
            save_intermediate_results=True
        )
        
        processor = BatchDocumentProcessor(config)
        results = await processor.process_documents(bmw_files)
        
        # Save VLM results
        for result in results:
            output_file = output_dir / f"{result.document_name}_vlm_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"✅ VLM analysis saved to {output_file}")

if __name__ == "__main__":
    print("BMW Document Processing Pipeline\n" + "="*50)
    asyncio.run(process_bmw_documents())