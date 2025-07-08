#!/usr/bin/env python3
"""
Debug script to examine Shikra model structure and MM projector conflict
"""

import torch
from transformers import LlamaForCausalLM
import sys

def examine_shikra_model():
    """Examine the Shikra model structure"""
    shikra_path = "umbrae/umbrae/model_weights/shikra-7b"
    
    print("🔍 Examining Shikra model structure:")
    print(f"   Path: {shikra_path}")
    
    try:
        # Try to load just the config first
        from transformers import LlamaConfig
        
        config_path = f"{shikra_path}/config.json"
        print(f"\n📋 Loading config from: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        print("📐 Config data:")
        for key, value in config_data.items():
            print(f"   {key}: {value}")
        
        print(f"\n🧪 Testing model loading approaches:")
        
        # Try loading with different parameters
        approaches = [
            {"approach": "Basic", "kwargs": {}},
            {"approach": "Ignore mismatched", "kwargs": {"ignore_mismatched_sizes": True}},
            {"approach": "No state dict", "kwargs": {"ignore_mismatched_sizes": True, "torch_dtype": torch.float16}},
        ]
        
        for i, test in enumerate(approaches, 1):
            print(f"\n   {i}. Testing {test['approach']} approach:")
            try:
                model = LlamaForCausalLM.from_pretrained(
                    shikra_path,
                    device_map="cpu",  # Load to CPU first
                    low_cpu_mem_usage=True,
                    **test['kwargs']
                )
                print(f"      ✅ Success! Model loaded")
                
                # Check if model has mm_projector
                if hasattr(model, 'mm_projector'):
                    print(f"      📍 Model has built-in mm_projector: {type(model.mm_projector)}")
                    if hasattr(model.mm_projector, 'weight'):
                        print(f"         Weight shape: {model.mm_projector.weight.shape}")
                    if hasattr(model.mm_projector, 'bias'):
                        print(f"         Bias shape: {model.mm_projector.bias.shape}")
                elif hasattr(model.model, 'mm_projector'):
                    print(f"      📍 Model.model has built-in mm_projector: {type(model.model.mm_projector)}")
                    if hasattr(model.model.mm_projector, 'weight'):
                        print(f"         Weight shape: {model.model.mm_projector.weight.shape}")
                    if hasattr(model.model.mm_projector, 'bias'):
                        print(f"         Bias shape: {model.model.mm_projector.bias.shape}")
                else:
                    print(f"      📍 No built-in mm_projector found")
                
                # Check model structure
                print(f"      📊 Model structure overview:")
                for name, module in model.named_children():
                    print(f"         {name}: {type(module)}")
                
                del model  # Free memory
                torch.cuda.empty_cache()
                break
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                
        # Check what's actually in the model files
        print(f"\n🗂️  Examining model files structure:")
        import os
        for file in os.listdir(shikra_path):
            if file.endswith('.safetensors'):
                print(f"   📄 {file}")
                
        # Try to peek at one safetensors file
        try:
            from safetensors import safe_open
            
            safetensors_file = f"{shikra_path}/model-00001-of-00003.safetensors"
            print(f"\n🔍 Examining {safetensors_file}:")
            
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                keys = f.keys()
                mm_projector_keys = [k for k in keys if 'mm_projector' in k.lower()]
                
                print(f"   Total keys: {len(list(keys))}")
                print(f"   MM projector related keys: {mm_projector_keys}")
                
                if mm_projector_keys:
                    for key in mm_projector_keys:
                        tensor = f.get_tensor(key)
                        print(f"      {key}: {tensor.shape} ({tensor.dtype})")
                        
        except Exception as e:
            print(f"   ❌ Could not examine safetensors: {e}")
            
    except Exception as e:
        print(f"❌ Failed to examine model: {e}")

if __name__ == "__main__":
    examine_shikra_model() 