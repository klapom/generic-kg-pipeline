#!/usr/bin/env python3
"""
Simple script to create sample documents for testing
"""

from pathlib import Path


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

Common Deep Learning Architectures:

1. Convolutional Neural Networks (CNNs)
   - Specialized for image processing
   - Uses convolution operations
   - Examples: AlexNet, VGG, ResNet

2. Recurrent Neural Networks (RNNs)
   - Designed for sequential data
   - Can process variable-length sequences
   - Examples: LSTM, GRU

3. Transformer Networks
   - Attention-based architecture
   - Parallel processing capability
   - Examples: BERT, GPT, T5

Training Deep Learning Models:

The training process involves:
1. Data preprocessing and augmentation
2. Model architecture design
3. Loss function selection
4. Optimization algorithm choice
5. Hyperparameter tuning
6. Regularization techniques

Challenges in Deep Learning:
- Requires large amounts of data
- Computationally expensive
- Overfitting risks
- Interpretability issues
- Ethical considerations

Future Directions:
- Few-shot learning
- Explainable AI
- Edge AI deployment
- Neuromorphic computing
- Quantum machine learning

Conclusion

Machine learning and deep learning continue to evolve rapidly, with new architectures and techniques being developed regularly. Understanding these fundamentals provides a solid foundation for exploring more advanced topics and applications in artificial intelligence.
""")
    
    print(f"📄 Created sample document: {sample_txt}")
    
    # Create a simple CSV-like document
    data_txt = input_dir / "sample_data.txt"
    with open(data_txt, 'w', encoding='utf-8') as f:
        f.write("""Research Paper Analysis

Title: Advanced Machine Learning Techniques for Natural Language Processing
Authors: Dr. Sarah Johnson, Prof. Michael Chen, Dr. Emma Rodriguez
Institution: Technical University of Computer Science
Year: 2023

Abstract:
This paper presents novel approaches to natural language processing using transformer-based architectures. We introduce a new attention mechanism that improves performance on various NLP tasks.

Key Findings:
- 15% improvement in BLEU scores for machine translation
- 22% reduction in training time compared to baseline models
- Better handling of long-range dependencies in text

Methodology:
The research employed a multi-phase approach:
1. Dataset preparation and preprocessing
2. Model architecture design
3. Experimental setup and training
4. Evaluation and comparison with baselines

Results:
Task                  | Baseline | Our Method | Improvement
Machine Translation   | 34.5     | 39.7       | +15.1%
Text Summarization    | 42.1     | 48.9       | +16.2%
Question Answering    | 78.2     | 85.6       | +9.5%

Technical Details:
- Model parameters: 175M
- Training data: 50GB of text
- GPU hours: 2,400
- Framework: PyTorch

Conclusion:
The proposed method demonstrates significant improvements across multiple NLP tasks while maintaining computational efficiency. Future work will focus on scaling to larger models and exploring multilingual capabilities.
""")
    
    print(f"📄 Created sample data document: {data_txt}")
    
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

## Available Sample Documents

- `sample_document.txt` - Overview of machine learning and deep learning
- `sample_data.txt` - Research paper analysis with structured data

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

## Testing the Current Implementation

To test the current implementation without external dependencies:

```bash
# Test with mock mode (no external services required)
python process_documents.py --mock

# Test with a specific file
python process_documents.py --file sample_document.txt --mock

# Create additional sample documents
python create_samples.py
```

The mock mode creates simple test documents and processes them through the chunking pipeline without requiring vLLM SmolDocling, Hochschul-LLM, or Qwen2.5-VL services.
""")
    
    print(f"📖 Created README: {readme_file}")
    
    # Create additional sample documents
    create_additional_samples(input_dir)


def create_additional_samples(input_dir):
    """Create additional sample documents for testing"""
    
    # Technical specification document
    spec_txt = input_dir / "technical_specification.txt"
    with open(spec_txt, 'w', encoding='utf-8') as f:
        f.write("""Generic Knowledge Graph Pipeline System - Technical Specification

Version: 1.0
Date: 2025-01-09

1. System Overview

The Generic Knowledge Graph Pipeline System is a comprehensive document processing platform that converts various document formats into structured knowledge graphs. The system supports multi-modal document analysis and implements context-aware chunking strategies.

2. Architecture Components

2.1 Document Parsing Layer
- vLLM SmolDocling: Advanced PDF parsing with GPU acceleration
- Multi-format parsers: DOCX, XLSX, PPTX support
- Visual element extraction: Images, charts, diagrams, tables

2.2 Content Processing Layer
- Context-aware chunking with inheritance
- Multi-modal content integration
- Qwen2.5-VL visual analysis

2.3 Knowledge Extraction Layer
- Hochschul-LLM integration via OpenAI API
- Triple extraction and validation
- Ontology-based knowledge representation

2.4 Storage Layer
- Apache Jena Fuseki (Triple Store)
- ChromaDB (Vector Store)
- Persistent storage and retrieval

3. Processing Pipeline

3.1 Document Ingestion
Input: PDF, DOCX, XLSX, PPTX, TXT files
Processing: Format-specific parsing and content extraction
Output: Structured document representation

3.2 Content Chunking
Strategy: Context inheritance vs. simple overlapping
Configuration: Token limits, boundary respect, async processing
Features: Context group formation, inheritance chains

3.3 Knowledge Extraction
Model: Qwen1.5-72B via Hochschul-LLM
Format: RDF triples using general ontology
Validation: Confidence scoring and quality assessment

3.4 Storage and Indexing
Triple Store: Structured knowledge graphs
Vector Store: Semantic search capabilities
Indexing: Efficient retrieval and querying

4. Configuration Management

4.1 Chunking Configuration (config/chunking.yaml)
- Document-specific strategies
- Token limits and overlap settings
- Context inheritance parameters
- Performance optimization settings

4.2 System Configuration (config/default.yaml)
- Service endpoints and credentials
- LLM model configurations
- Storage backend settings
- Domain-specific ontologies

5. API Endpoints

5.1 Document Management
- POST /documents/upload - Upload documents
- GET /documents/{id} - Retrieve document info
- DELETE /documents/{id} - Remove documents

5.2 Processing Control
- POST /documents/{id}/process - Start processing
- GET /documents/{id}/status - Check processing status
- GET /documents/{id}/results - Retrieve results

5.3 Knowledge Graph Operations
- GET /knowledge/entities - Query entities
- GET /knowledge/relationships - Query relationships
- POST /knowledge/query - SPARQL queries

6. Performance Specifications

6.1 Throughput
- PDF processing: 5-10 pages/minute (depends on complexity)
- Text documents: 100-500 KB/minute
- Concurrent processing: Up to 10 documents

6.2 Scalability
- Horizontal scaling: Multiple worker instances
- GPU utilization: Dedicated GPU for vLLM SmolDocling
- Memory requirements: 16GB+ for large documents

6.3 Quality Metrics
- Triple extraction accuracy: >85%
- Context preservation: >90%
- Visual element recognition: >80%

7. Dependencies and Requirements

7.1 External Services
- vLLM SmolDocling service (GPU required)
- Hochschul-LLM API endpoint
- Qwen2.5-VL service (via Hochschul-LLM)

7.2 Storage Systems
- Apache Jena Fuseki server
- ChromaDB instance
- File system for temporary storage

7.3 Python Dependencies
- FastAPI for REST API
- AsyncIO for concurrent processing
- Pydantic for data validation
- PyTorch for model integration

8. Security Considerations

8.1 API Security
- Authentication via API keys
- Input validation and sanitization
- Rate limiting and abuse prevention

8.2 Data Protection
- Secure document storage
- Encryption for sensitive data
- Access control and auditing

9. Monitoring and Logging

9.1 Performance Monitoring
- Processing time tracking
- Error rate monitoring
- Resource utilization metrics

9.2 Audit Trail
- Document processing history
- User activity logging
- System event recording

10. Future Enhancements

10.1 Planned Features
- Multi-language support
- Advanced visualization tools
- Batch processing optimization
- Real-time processing capabilities

10.2 Research Directions
- Improved context understanding
- Better visual element analysis
- Domain-specific ontology learning
- Automated quality assessment
""")
    
    print(f"📄 Created technical specification: {spec_txt}")
    
    # Tutorial document
    tutorial_txt = input_dir / "user_tutorial.txt"
    with open(tutorial_txt, 'w', encoding='utf-8') as f:
        f.write("""User Tutorial: Getting Started with Knowledge Graph Pipeline

Welcome to the Generic Knowledge Graph Pipeline System! This tutorial will guide you through the process of converting your documents into structured knowledge graphs.

Step 1: Prepare Your Documents

Supported formats:
- PDF documents (research papers, reports, manuals)
- Word documents (.docx) with images and tables
- Excel spreadsheets (.xlsx) with charts and data
- PowerPoint presentations (.pptx) with slides and visuals
- Plain text files (.txt)

Best practices:
- Ensure documents are not password-protected
- Use clear headings and structure
- Include alt-text for images when possible
- Keep file sizes under 50MB for optimal performance

Step 2: Upload Documents

Using the Web Interface:
1. Navigate to the upload page
2. Select your documents
3. Click "Upload" to transfer files
4. Wait for upload confirmation

Using the API:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Step 3: Start Processing

Web Interface:
1. Go to the documents list
2. Click "Process" next to your document
3. Select processing options
4. Monitor progress in real-time

API Command:
```bash
curl -X POST "http://localhost:8000/documents/{id}/process" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Step 4: Monitor Progress

The system provides real-time updates on:
- Document parsing progress
- Chunking status
- Knowledge extraction phase
- Storage and indexing completion

Status indicators:
- 🔄 Processing: Document is being processed
- ✅ Complete: Processing finished successfully
- ❌ Error: Processing failed (check logs)
- ⏸️ Paused: Processing temporarily stopped

Step 5: Review Results

Once processing is complete, you can:

View Knowledge Graph:
- Browse extracted entities
- Explore relationships
- Filter by categories
- Export in various formats

Download Results:
- RDF/Turtle format for technical use
- JSON for application integration
- CSV for spreadsheet analysis
- Visual graph formats (GraphML, DOT)

Step 6: Query Your Knowledge

SPARQL Queries:
```sparql
PREFIX kg: <https://generic-kg-pipeline.org/ontology/>

SELECT ?entity ?type ?description
WHERE {
  ?entity a kg:Entity .
  ?entity kg:hasName ?name .
  ?entity kg:hasDescription ?description .
  FILTER(CONTAINS(?name, "machine learning"))
}
```

Natural Language Queries:
- "What are the main concepts in this document?"
- "Show me all people mentioned in the text"
- "Find relationships between technologies"

Step 7: Advanced Features

Context Inheritance:
- Maintains context across document chunks
- Improves knowledge extraction accuracy
- Configurable inheritance depth

Multi-Modal Analysis:
- Analyzes images, charts, and diagrams
- Extracts text from visual elements
- Connects visual and textual information

Batch Processing:
- Process multiple documents simultaneously
- Efficient resource utilization
- Progress tracking for large batches

Step 8: Troubleshooting

Common Issues:

Document Not Processing:
- Check file format compatibility
- Verify file is not corrupted
- Ensure sufficient system resources

Poor Extraction Quality:
- Review document structure
- Check for scan quality (PDFs)
- Validate input language support

Performance Issues:
- Monitor system resources
- Reduce concurrent processing
- Optimize document size

Getting Help:
- Check the FAQ section
- Review error logs
- Contact support team
- Submit bug reports

Step 9: Best Practices

Document Preparation:
- Use consistent formatting
- Include descriptive headings
- Provide context for abbreviations
- Structure information logically

Processing Optimization:
- Process similar documents together
- Use appropriate chunking strategies
- Monitor resource usage
- Regular system maintenance

Knowledge Graph Quality:
- Review extracted entities
- Validate relationships
- Provide feedback for improvements
- Maintain ontology consistency

Step 10: Next Steps

After mastering the basics:
- Explore advanced query techniques
- Integrate with external applications
- Customize ontologies for your domain
- Contribute to system improvements

Additional Resources:
- API documentation
- Video tutorials
- Community forums
- Technical specifications

Happy knowledge graphing!
""")
    
    print(f"📄 Created user tutorial: {tutorial_txt}")


if __name__ == "__main__":
    create_sample_documents()
    print("\n✅ Sample documents created successfully!")
    print("📁 Check the data/input/ directory for the files")
    print("🔄 Run 'python process_documents.py --mock' to test processing")