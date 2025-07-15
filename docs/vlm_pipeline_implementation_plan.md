# VLM Pipeline Implementation Plan

## Overview
Implementation of a resource-efficient two-stage VLM pipeline with confidence-based fallback mechanism.

## Research Findings

### Confidence Score Best Practices (2024)
1. **Dynamic Thresholds**: 0.7-0.85 typical for fallback decisions
2. **Precision-Recall Analysis**: Essential for threshold optimization
3. **Graceful Degradation**: Multi-tiered fallback strategies
4. **Context-Aware Thresholds**: Consider error costs per use case

### Key Success Factors
- Small models (3-7B) now achieve performance of 540B models from 2022
- Mixture-of-Experts architectures reduce hallucinations
- 78% of organizations using AI in production (2024)

## Implementation Plan

### Phase 1: Two-Stage VLM Strategy
**Goal**: Implement sequential model loading with Qwen → Pixtral workflow

#### 1.1 Model Manager Class
```python
class VLMModelManager:
    """Manages model loading/unloading for memory efficiency"""
    - load_qwen()
    - load_pixtral()
    - cleanup()
    - get_current_model()
```

#### 1.2 Document Classifier
```python
class DocumentElementClassifier:
    """Classifies images as diagram/table/text/general"""
    - is_diagram(image_path) → bool
    - detect_element_type(image_data) → VisualElementType
    - requires_high_precision(element_type) → bool
```

#### 1.3 Two-Stage Processor
```python
class TwoStageVLMProcessor:
    """Orchestrates two-stage processing"""
    - process_batch(documents)
    - stage1_qwen_processing(non_diagrams)
    - stage2_pixtral_processing(diagrams)
```

### Phase 2: Batch Processing System
**Goal**: Process multiple documents efficiently with batching

#### 2.1 Batch Manager
```python
class BatchDocumentProcessor:
    """Handles batch processing with optimal sizing"""
    - optimal_batch_size = 4-8 images
    - process_document_batch(pdf_paths)
    - handle_memory_constraints()
```

#### 2.2 Performance Optimizations
- Implement dynamic batch sizing based on available memory
- Use Flash Attention 2 for Qwen2.5-VL
- Configure optimal resolution per document type

### Phase 3: Confidence-Based Fallback
**Goal**: Intelligent model switching based on confidence scores

#### 3.1 Confidence Evaluator
```python
class ConfidenceEvaluator:
    """Evaluates and decides on fallback needs"""
    - confidence_threshold = 0.85 (configurable)
    - needs_fallback(result) → bool
    - calculate_aggregate_confidence(results) → float
```

#### 3.2 Fallback Strategy
```python
class FallbackStrategy:
    """Implements fallback logic"""
    - Strategy 1: Low confidence → Pixtral
    - Strategy 2: Parsing errors → Retry with different params
    - Strategy 3: Critical documents → Always use both models
```

## Potential Errors & Mitigations

### 1. Memory Management
**Error**: GPU OOM when switching models
**Mitigation**: 
- Explicit cleanup with torch.cuda.empty_cache()
- Model deletion before loading new one
- Monitor GPU memory usage

### 2. Batch Processing
**Error**: Variable image sizes causing batch errors
**Mitigation**:
- Dynamic padding
- Separate batches by image size
- Fallback to single processing

### 3. Confidence Calibration
**Error**: Overconfident wrong predictions (hallucinations)
**Mitigation**:
- Cross-validation with multiple prompts
- Temperature adjustment (0.2-0.3)
- Ensemble voting for critical decisions

### 4. Model Loading
**Error**: Slow model switching
**Mitigation**:
- Keep processor loaded
- Use 8-bit quantization
- Pre-compile with torch.compile() if available

## Testing Strategy

### Unit Tests
1. Test model manager switching
2. Test document classifier accuracy
3. Test confidence evaluator thresholds

### Integration Tests
1. Process BMW X5 document with full pipeline
2. Verify memory usage stays under limits
3. Measure end-to-end performance

### Performance Benchmarks
- Target: < 5s per image with Qwen
- Target: < 30s per diagram with Pixtral
- Target: < 100s for 17-page document

## Success Criteria
1. ✅ All existing functionality preserved
2. ✅ Memory usage < 16GB GPU RAM
3. ✅ Batch processing 2-3x faster
4. ✅ Confidence-based fallback improves accuracy
5. ✅ No degradation in current performance

## Risk Assessment
- **Low Risk**: Model manager implementation
- **Medium Risk**: Batch processing edge cases
- **Low Risk**: Confidence threshold tuning

## Implementation Order
1. VLMModelManager (foundation)
2. DocumentElementClassifier (enables smart routing)
3. TwoStageVLMProcessor (core logic)
4. BatchDocumentProcessor (performance)
5. ConfidenceEvaluator + FallbackStrategy (quality)

## Rollback Plan
- All new code in separate modules
- Existing clients unchanged
- Feature flags for gradual rollout