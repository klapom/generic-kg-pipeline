#!/usr/bin/env python3
"""
Test vLLM with smallest possible model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vllm_availability():
    """Test if vLLM is available"""
    try:
        import vllm
        print(f"✅ vLLM available: {vllm.__version__}")
        return True
    except ImportError as e:
        print(f"❌ vLLM not available: {e}")
        return False

def test_simple_model():
    """Test with a simple model"""
    from vllm import LLM, SamplingParams
    
    print("🧪 Testing simple vLLM model...")
    
    try:
        # Use a very small model for testing
        model_name = "microsoft/DialoGPT-small"
        
        # Create LLM with minimal settings
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.3,  # Very conservative
            max_model_len=512,
            trust_remote_code=True
        )
        
        # Test generation
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50)
        
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"✅ Generated text: {output.outputs[0].text}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing simple model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing vLLM system...")
    
    if not test_vllm_availability():
        exit(1)
    
    if not test_simple_model():
        exit(1)
    
    print("✅ Basic vLLM test completed successfully!")