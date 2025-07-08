#!/usr/bin/env python3
"""
Debug script to examine MM projector loading issue
"""

import torch
import sys

def examine_mm_projector():
    """Examine the MM projector file structure"""
    adapter_path = "umbrae/umbrae/model_weights/mm_projector.bin"
    
    print("🔍 Examining MM Projector file:")
    print(f"   Path: {adapter_path}")
    
    try:
        # Load the file
        mm_projector_weights = torch.load(adapter_path, map_location='cpu')
        
        print(f"\n📋 File structure:")
        print(f"   Type: {type(mm_projector_weights)}")
        
        if isinstance(mm_projector_weights, dict):
            print(f"   Keys: {list(mm_projector_weights.keys())}")
            
            print(f"\n📐 Weight shapes:")
            for key, value in mm_projector_weights.items():
                if torch.is_tensor(value):
                    print(f"   {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"   {key}: {type(value)} - {value}")
        
        # Test what a standard Linear layer expects
        print(f"\n🧪 Standard Linear(1024, 4096) layer structure:")
        test_layer = torch.nn.Linear(1024, 4096)
        print(f"   Expected keys: {list(test_layer.state_dict().keys())}")
        for key, value in test_layer.state_dict().items():
            print(f"   {key}: {value.shape} ({value.dtype})")
        
        # Test direct key mapping
        print(f"\n🔧 Testing key mappings:")
        
        # Strategy 1: Direct mapping
        if 'model.mm_projector.weight' in mm_projector_weights and 'model.mm_projector.bias' in mm_projector_weights:
            print("   ✓ Direct keys found")
            state_dict_1 = {
                'weight': mm_projector_weights['model.mm_projector.weight'],
                'bias': mm_projector_weights['model.mm_projector.bias']
            }
            print(f"   Strategy 1 shapes:")
            for k, v in state_dict_1.items():
                print(f"     {k}: {v.shape}")
        
        # Strategy 2: Alternative key patterns
        weight_keys = [k for k in mm_projector_weights.keys() if 'weight' in k.lower()]
        bias_keys = [k for k in mm_projector_weights.keys() if 'bias' in k.lower()]
        
        print(f"   Weight keys found: {weight_keys}")
        print(f"   Bias keys found: {bias_keys}")
        
        # Test loading into actual layer
        print(f"\n🧪 Testing actual loading:")
        test_mm_projector = torch.nn.Linear(1024, 4096)
        
        try:
            # Try the mapping that's currently in the code
            if 'model.mm_projector.weight' in mm_projector_weights:
                state_dict = {
                    'weight': mm_projector_weights['model.mm_projector.weight'],
                    'bias': mm_projector_weights['model.mm_projector.bias']
                }
                test_mm_projector.load_state_dict(state_dict)
                print("   ✅ Loading successful!")
                
                # Test the layer works
                test_input = torch.randn(1, 256, 1024)
                test_output = test_mm_projector(test_input)
                print(f"   ✅ Forward pass successful: {test_input.shape} -> {test_output.shape}")
                
            else:
                print("   ❌ Expected keys not found")
                
        except Exception as e:
            print(f"   ❌ Loading failed: {e}")
            
        return mm_projector_weights
            
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return None

if __name__ == "__main__":
    examine_mm_projector() 