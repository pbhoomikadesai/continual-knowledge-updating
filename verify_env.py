"""
Phase 0: Environment Verification Script
Loads Qwen/Qwen2.5-0.5B-Instruct, checks CUDA/CPU availability, and generates a test response.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def verify_environment():
    print("=" * 60)
    print("Phase 0: Environment & Hardware Verification")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device.upper()}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Note: Running on CPU (CUDA unavailable). Fine for dataset/RAG setup; use GPU for training.")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\nLoading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cpu":
            model = model.to("cpu")
            
        print("Model and Tokenizer loaded successfully!")
        
        # Test generation
        prompt = "Question: What is the capital of France?\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.1,
                do_sample=False
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest Prompt:\n{prompt}")
        print(f"Test Generation:\n{response}")
        print("\n[SUCCESS] Phase 0 Verification Complete!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Environment verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_environment()
