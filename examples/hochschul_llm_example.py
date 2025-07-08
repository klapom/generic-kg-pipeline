"""Example usage of Hochschul-LLM client for triple extraction"""

import asyncio
import logging
from typing import List

from core.clients.hochschul_llm import HochschulLLMClient, TripleExtractionConfig, Triple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def single_extraction_example():
    """Example: Extract triples from a single text"""
    print("=== Single Text Triple Extraction Example ===")
    
    # Sample text for extraction
    text = """
    Dr. Sarah Johnson is a professor of computer science at Stanford University.
    She received her PhD from MIT in 2010 and specializes in machine learning
    and artificial intelligence. Her research has been published in Nature and
    Science journals. Stanford University is located in California and was 
    founded in 1885 by Leland Stanford.
    """
    
    try:
        async with HochschulLLMClient() as client:
            print(f"📄 Extracting triples from text ({len(text)} chars)")
            
            # Extract triples
            result = await client.extract_triples(text)
            
            if result.success:
                print(f"✅ Extraction successful!")
                print(f"   📊 Triples extracted: {result.triple_count}")
                print(f"   ⏱️  Processing time: {result.processing_time_seconds:.2f}s")
                print(f"   🤖 Model used: {result.model_used}")
                print(f"   📈 Average confidence: {result.average_confidence:.2f}")
                
                # Display extracted triples
                print("\n📋 Extracted Triples:")
                for i, triple in enumerate(result.triples[:5]):  # Show first 5
                    print(f"   {i+1}. {triple.subject} --[{triple.predicate}]--> {triple.object}")
                    print(f"      Confidence: {triple.confidence:.2f}")
                    if triple.context:
                        print(f"      Context: {triple.context[:100]}...")
                    print()
                
                # Show metadata
                if result.metadata:
                    print("🔍 Extraction Metadata:")
                    for key, value in result.metadata.items():
                        print(f"   {key}: {value}")
                
            else:
                print(f"❌ Extraction failed: {result.error_message}")
                
    except Exception as e:
        print(f"❌ Error: {e}")


async def batch_extraction_example():
    """Example: Extract triples from multiple text chunks"""
    print("\n=== Batch Triple Extraction Example ===")
    
    # Multiple text chunks
    text_chunks = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "The company is headquartered in Cupertino, California.",
        "Tim Cook became the CEO of Apple in 2011, succeeding Steve Jobs.",
        "Apple develops and sells consumer electronics, software, and online services.",
        "The iPhone was first released in 2007 and revolutionized the smartphone industry."
    ]
    
    try:
        async with HochschulLLMClient() as client:
            print(f"📄 Batch extracting from {len(text_chunks)} text chunks")
            
            # Extract triples from all chunks
            results = await client.extract_triples_batch(text_chunks)
            
            # Summary
            successful = sum(1 for r in results if r.success)
            total_triples = sum(r.triple_count for r in results)
            total_time = sum(r.processing_time_seconds for r in results)
            
            print(f"✅ Batch extraction completed!")
            print(f"   📊 Successful chunks: {successful}/{len(results)}")
            print(f"   🔢 Total triples: {total_triples}")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   📈 Average time per chunk: {total_time/len(results):.2f}s")
            
            # Show results for each chunk
            print("\n📋 Results by Chunk:")
            for i, (chunk, result) in enumerate(zip(text_chunks, results)):
                if result.success:
                    print(f"   {i+1}. ✅ {result.triple_count} triples (conf: {result.average_confidence:.2f})")
                    print(f"      Text: {chunk[:60]}...")
                    
                    # Show top triples
                    for j, triple in enumerate(result.triples[:2]):
                        print(f"         • {triple.subject} → {triple.predicate} → {triple.object}")
                else:
                    print(f"   {i+1}. ❌ Failed: {result.error_message}")
                    print(f"      Text: {chunk[:60]}...")
                print()
                    
    except Exception as e:
        print(f"❌ Batch extraction error: {e}")


async def domain_specific_extraction_example():
    """Example: Domain-specific extraction with context and ontology hints"""
    print("\n=== Domain-Specific Extraction Example ===")
    
    # Scientific paper abstract
    scientific_text = """
    This study investigates the effectiveness of transformer-based neural networks
    for natural language processing tasks. We trained a BERT model on a dataset
    of 50,000 scientific papers. The model achieved 92% accuracy on text classification
    and 89% on named entity recognition. The research was conducted at MIT by
    Dr. Maria Rodriguez and her team. The results were published in the Journal
    of Machine Learning Research in 2023.
    """
    
    try:
        async with HochschulLLMClient() as client:
            print("📄 Extracting with domain context and ontology hints")
            
            # Extract with domain-specific context
            result = await client.extract_triples(
                scientific_text,
                domain_context="scientific research and machine learning",
                ontology_hints=[
                    "hasAuthor", "publishedIn", "achievedAccuracy", "conducteAt",
                    "investigates", "usesModel", "trainedOn", "hasMetric"
                ]
            )
            
            if result.success:
                print(f"✅ Domain-specific extraction successful!")
                print(f"   📊 Triples: {result.triple_count}")
                print(f"   🎯 Domain detected: {result.metadata.get('domain_detected', 'N/A')}")
                
                # Group triples by predicate type
                predicate_groups = {}
                for triple in result.triples:
                    pred = triple.predicate
                    if pred not in predicate_groups:
                        predicate_groups[pred] = []
                    predicate_groups[pred].append(triple)
                
                print("\n📊 Triples by Relationship Type:")
                for predicate, triples in predicate_groups.items():
                    print(f"   {predicate}: {len(triples)} triples")
                    for triple in triples[:2]:  # Show first 2
                        print(f"      • {triple.subject} → {triple.object}")
                
            else:
                print(f"❌ Domain extraction failed: {result.error_message}")
                
    except Exception as e:
        print(f"❌ Domain extraction error: {e}")


async def validation_example():
    """Example: Validate extracted triples for quality"""
    print("\n=== Triple Validation Example ===")
    
    # Create some example triples with various quality levels
    triples = [
        Triple("Einstein", "proposed", "Theory of Relativity", 0.95),
        Triple("", "hasAge", "30", 0.80),  # Empty subject
        Triple("Something", "relatedTo", "thing", 0.85),  # Generic terms
        Triple("Berlin", "locatedIn", "Germany", 0.60),  # Low confidence
        Triple("Python", "isA", "programming language", 0.90),
    ]
    
    try:
        async with HochschulLLMClient() as client:
            print(f"🔍 Validating {len(triples)} triples")
            
            validation_report = await client.validate_triples(triples)
            
            print(f"📊 Validation Results:")
            print(f"   Total triples: {validation_report['total_triples']}")
            print(f"   Valid triples: {validation_report['valid_triples']}")
            print(f"   Quality score: {validation_report['quality_score']:.2f}")
            
            if validation_report['issues']:
                print(f"\n⚠️  Issues Found:")
                for issue in validation_report['issues'][:3]:  # Show first 3
                    triple_info = issue['triple']
                    print(f"   • {triple_info['subject']} → {triple_info['predicate']} → {triple_info['object']}")
                    print(f"     Issues: {', '.join(issue['issues'])}")
            
            if validation_report['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in validation_report['recommendations']:
                    print(f"   • {rec}")
                    
    except Exception as e:
        print(f"❌ Validation error: {e}")


async def health_check_example():
    """Example: Check Hochschul-LLM service health"""
    print("\n=== Health Check Example ===")
    
    try:
        async with HochschulLLMClient() as client:
            health = await client.health_check()
            
            print(f"🏥 Service status: {health['status']}")
            print(f"🔗 Endpoint: {health['endpoint']}")
            print(f"🤖 Model: {health['model']}")
            
            if health['status'] == 'healthy':
                print(f"⚡ Response time: {health.get('response_time_ms', 0):.2f}ms")
                print(f"✅ Test response: {health.get('test_response', 'N/A')}")
                
                model_info = health.get('model_info', {})
                if model_info:
                    print(f"📊 Model info: {model_info}")
            else:
                print(f"❌ Error: {health.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")


async def configuration_example():
    """Example: Different extraction configurations"""
    print("\n=== Configuration Example ===")
    
    # Different configurations for different use cases
    configs = {
        "high_precision": TripleExtractionConfig(
            temperature=0.0,
            confidence_threshold=0.9,
            max_tokens=2000
        ),
        "balanced": TripleExtractionConfig(
            temperature=0.1,
            confidence_threshold=0.7,
            max_tokens=4000
        ),
        "exploratory": TripleExtractionConfig(
            temperature=0.3,
            confidence_threshold=0.5,
            max_tokens=6000
        )
    }
    
    sample_text = "Steve Jobs co-founded Apple with Steve Wozniak in 1976."
    
    for config_name, config in configs.items():
        print(f"\n📋 Configuration: {config_name}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Confidence threshold: {config.confidence_threshold}")
        print(f"   Max tokens: {config.max_tokens}")
        
        try:
            async with HochschulLLMClient(config) as client:
                result = await client.extract_triples(sample_text)
                
                if result.success:
                    high_conf_triples = result.filter_by_confidence(config.confidence_threshold)
                    print(f"   ✅ Extracted: {result.triple_count} total, {len(high_conf_triples)} high-confidence")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
                    
        except Exception as e:
            print(f"   ❌ Error with {config_name}: {e}")


async def main():
    """Run all examples"""
    print("🚀 Hochschul-LLM Client Examples")
    print("=" * 50)
    
    # Run examples
    await health_check_example()
    await configuration_example()
    await single_extraction_example()
    await batch_extraction_example()
    await domain_specific_extraction_example()
    await validation_example()
    
    print("\n✨ Examples completed!")
    print("\n💡 To use with real Hochschul-LLM:")
    print("   1. Configure HOCHSCHUL_LLM_ENDPOINT and HOCHSCHUL_LLM_API_KEY")
    print("   2. Make sure the OpenAI-compatible API is accessible")
    print("   3. Adjust configuration parameters as needed")


if __name__ == "__main__":
    asyncio.run(main())