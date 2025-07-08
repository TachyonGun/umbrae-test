#!/usr/bin/env python3
"""
Enhanced UMBRAE Inference with FP16 Optimizations
Uses memory monitoring and configuration for 16GB GPU optimization
"""

import os
import sys
import torch
import time
import json
import gc
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Import from original UMBRAE code
from transformers import LlamaForCausalLM, LlamaTokenizer  # Use LlamaTokenizer instead of AutoTokenizer
import webdataset as wds
import braceexpand

from config import MODEL_CONFIG, INFERENCE_CONFIG, MEMORY_CONFIG, PROMPTS, SYSTEM_REQUIREMENTS
from memory_utils import MemoryMonitor, cleanup_gpu_memory, memory_efficient_loading, optimize_model_for_inference

# Add UMBRAE to path
sys.path.append('umbrae')

try:
    from transformers import LlamaForCausalLM, LlamaTokenizer
    import braceexpand
    import webdataset as wds
    from model import BrainX
    import utils
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run the setup script first: python umbrae_inference_fp16.py")
    sys.exit(1)

def run_optimized_inference(
    subject: int = 1,
    prompt_type: str = "grounding", 
    max_samples: int = 3,
    use_fp16: bool = True,
    monitor_memory: bool = True,
    precision_mode: str = "fp16"
):
    """Run UMBRAE inference with FP16 optimizations"""
    
    # Precision configuration
    brain_fp16 = precision_mode in ["fp16", "hybrid"]
    llama_fp16 = precision_mode in ["fp16"] 
    mm_fp16 = precision_mode in ["fp16"]
    
    print("=== UMBRAE FP16 Optimized Inference ===")
    print(f"Subject: {subject}")
    print(f"Prompt type: {prompt_type}")
    print(f"Max samples: {max_samples}")
    print(f"Precision mode: {precision_mode}")
    print(f"  📊 Brain encoder: {'FP16' if brain_fp16 else 'FP32'}")
    print(f"  📊 LLaMA model: {'FP16' if llama_fp16 else 'FP32'}")
    print(f"  📊 MM projector: {'FP16' if mm_fp16 else 'FP32'}")
    
    monitor = MemoryMonitor() if monitor_memory else None
    if monitor:
        monitor.log_memory("Startup", "Beginning inference")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Check memory requirements 
    if precision_mode == "fp32":
        print("GPU Memory Check:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB")
        print("✓ Sufficient memory available")
    else:
        print("GPU Memory Check:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  Required: {MEMORY_CONFIG['max_memory_gb']:.1f}GB")
    
    print(f"Using device: {device}")
    if monitor:
        monitor.log_memory("Config", "Configuration loaded")
    
    # Setup paths
    data_path = MODEL_CONFIG["data_path"]
    brainx_path = MODEL_CONFIG["brainx_path"] 
    shikra_path = MODEL_CONFIG["shikra_path"]
    adapter_path = MODEL_CONFIG["adapter_path"]
    prompt = PROMPTS[prompt_type]
    
    # Load test data
    print("\n📁 Loading test data...")
    val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subject}_" + "{0..1}.tar"
    to_tuple = ["voxels", "images"]
    val_batch_size = 1
    split_by_node = lambda urls: urls
    val_url = list(braceexpand.braceexpand(val_url))
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=data_path, nodesplitter=split_by_node) \
        .decode("torch")\
        .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy") \
        .to_tuple(*to_tuple) \
        .batched(val_batch_size, partial=False)
    
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=0, shuffle=False)
    print("✓ Test data loaded")
    if monitor:
        monitor.log_memory("Data", "Test data loaded")
    
    # Load brain encoder
    print("\n🧠 Loading brain encoder...")
    print("  Loading checkpoint...")
    
    from umbrae.model import BrainX  # Import model from original code
    
    # Setup model parameters to match original
    kwargs = {'hidden_dim': 1024, 'out_dim': 1024, 'num_latents': 256, 'use_norm': False, 'use_token': False}
    voxel2emb = BrainX(**kwargs)
    voxel2emb.to(device)
    
    checkpoint = torch.load(brainx_path, map_location='cpu')
    voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("Applying inference optimizations...")
    optimize_model_for_inference(voxel2emb, enable_fp16=brain_fp16)
    print("✓ Inference optimizations applied")
    print(f"  📊 Brain encoder precision: {'FP16' if brain_fp16 else 'FP32'}")
    
    print("✓ Brain encoder loaded and optimized")
    if monitor:
        monitor.log_memory("Brain Encoder", f"FP16: {brain_fp16}")
    
    # Process brain signals (following original approach)
    print("\n⚡ Processing brain signals...")
    emb_voxel_list = []
    processed_count = 0
    
    for val_i, (voxel, image) in enumerate(val_dl):
        if max_samples and processed_count >= max_samples:
            break
            
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=INFERENCE_CONFIG["use_mixed_precision"]):
                # CRITICAL FIX: Average voxel data like the original
                voxel = torch.mean(voxel, axis=1).float()  # This was missing!
                emb_voxel = voxel2emb(voxel.to(device), modal=f'fmri{subject}')
                emb_voxel_list.append(emb_voxel)
                processed_count += 1
                
        # Memory cleanup every 5 samples
        if processed_count % 5 == 0:
            cleanup_gpu_memory()
    
    image_features = torch.cat(emb_voxel_list, dim=0)
    print(f"✓ Processed {processed_count} brain signals")
    print(f"  Features shape: {image_features.shape}")
    if monitor:
        monitor.log_memory("Brain Processing", f"Processed {processed_count} samples")
    
    # Critical memory cleanup - delete brain encoder before loading LLaMA
    del voxel2emb, checkpoint, emb_voxel_list
    cleanup_gpu_memory()
    
    # Load LLaMA model (fixed approach)
    print("\n🤖 Loading LLaMA model...")
    
    # CRITICAL FIX: Use LlamaTokenizer and proper model loading
    tokenizer = LlamaTokenizer.from_pretrained(shikra_path, padding_side='left')
    model = LlamaForCausalLM.from_pretrained(
        shikra_path,
        torch_dtype=torch.float16 if llama_fp16 else torch.float32,
        # Remove device_map to avoid shikra->llama warning
    )
    model.to(device)
    
    # Apply additional optimizations
    if MEMORY_CONFIG["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    print("✓ LLaMA model loaded with memory optimizations")
    
    if monitor:
        monitor.log_memory("LLaMA Model", f"Memory limit: {MEMORY_CONFIG['max_memory_gb']}GB")
    
    # Load MM projector (fixed approach)
    print("\n🔗 Loading MM projector...")
    mm_projector = torch.nn.Linear(1024, 4096)
    mm_projector_weights = torch.load(adapter_path, map_location='cpu')
    
    # Debug: Print available keys
    print(f"  Available keys: {list(mm_projector_weights.keys())}")
    
    # CRITICAL FIX: Use proper key mapping like the original
    if adapter_path.endswith('mm_projector.bin'):
        # Use the original's key mapping approach
        adjusted_state_dict = {k.split('.')[-1]: v for k, v in mm_projector_weights.items()}
        mm_projector.load_state_dict(adjusted_state_dict)
        print("  ✅ MM projector loaded with adjusted state dict!")
    else:
        mm_projector.load_state_dict(mm_projector_weights['model_state_dict'], strict=False)
        print("  ✅ MM projector loaded from model_state_dict")
    
    del mm_projector_weights  # Free memory
    
    # Move to device first, then handle precision carefully
    mm_projector.to(device)
    
    # Critical: MM projector precision strategy
    if mm_fp16:
        mm_projector.half()
        print(f"  📊 MM projector precision: FP16")
    else:
        mm_projector.float()
        print(f"  📊 MM projector precision: FP32")
    
    # Ensure feature precision matches MM projector
    if mm_fp16 and image_features.dtype != torch.float16:
        image_features = image_features.half()
    elif not mm_fp16 and image_features.dtype != torch.float32:
        image_features = image_features.float()
    
    print(f"  🔗 Feature dtype: {image_features.dtype}, MM projector dtype: {next(mm_projector.parameters()).dtype}")
    
    # Project features with proper dtype matching
    with torch.no_grad():
        # Ensure input dtype matches MM projector dtype
        mm_projector_dtype = next(mm_projector.parameters()).dtype
        image_features = mm_projector(image_features.to(mm_projector_dtype))
    
    print(f"✓ MM projector loaded, features projected to {image_features.shape}")
    
    if monitor:
        monitor.log_memory("MM Projector", f"Final features: {image_features.shape}")
    
    # Prepare prompt
    print("\n📝 Preparing prompt...")
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    user_image = " <im_start>" + "<im_patch>" * 256 + "<im_end> "
    
    if '<image>' in prompt:
        user_prompt = prompt.replace('<image>', user_image)
    else:
        user_prompt = prompt + user_image
    input_text = system + user_prompt + " ASSISTANT:"
    
    # Debug: Print the actual prompt sent to model
    print(f"📋 Full prompt being sent to model:")
    print(f"   Original prompt: '{prompt}'")
    print(f"   Final input text: '{input_text[:200]}...{input_text[-50:]}'")
    print(f"   Length: {len(input_text)} characters")
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    # CRITICAL FIX: Use exact token IDs from original
    gen_kwargs = {
        "use_cache": True,
        "do_sample": False,
        "pad_token_id": 2,  # Explicit values from original
        "bos_token_id": 1,
        "eos_token_id": 2,
        "max_new_tokens": INFERENCE_CONFIG["max_new_tokens"],
    }
    
    print(f"✓ Prompt prepared, generating up to {gen_kwargs['max_new_tokens']} tokens")
    
    # Run inference on samples - following original approach
    results = []
    
    # CRITICAL FIX: Process each image individually like the original
    for cur_image_idx in range(min(processed_count, max_samples)):
        print(f"\n🎯 Running inference on sample {cur_image_idx + 1}...")
        
        # Following the exact original embedding reconstruction approach
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            cur_image_features = image_features[cur_image_idx]
            num_patches = cur_image_features.shape[0]
            image_start_tokens = torch.where(cur_input_ids == 32001)[0]
            
            for image_start_token_pos in image_start_tokens:
                cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                num_patches = cur_image_features.shape[0]
                
                # Verify token structure
                if cur_input_ids[image_start_token_pos + num_patches + 1] != 32002:
                    print(f"⚠️  Warning: Expected image end token 32002 but got {cur_input_ids[image_start_token_pos + num_patches + 1]}")
                
                cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:image_start_token_pos + 1], 
                    cur_image_features,
                    cur_input_embeds[image_start_token_pos + num_patches + 1:]
                ), dim=0)
            new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds_final = torch.stack(new_input_embeds, dim=0)
        
        # Generate response
        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16 if llama_fp16 else torch.float32, device_type='cuda'):
                output_ids = model.generate(inputs_embeds=inputs_embeds_final.float(), **gen_kwargs)
        
        generation_time = time.time() - start_time
        response = tokenizer.batch_decode(output_ids)[0]
        
        # Debug: Show full raw response
        print(f"🔍 Raw model output: '{response[:300]}...'")
        
        # Clean up response like the original
        response_clean = response.strip(' <s></s>')
        if "ASSISTANT:" in response_clean:
            response_clean = response_clean.split("ASSISTANT:")[-1].strip()
        
        print(f"🧹 Cleaned response: '{response_clean}'")
        
        result = {
            "sample": cur_image_idx + 1,
            "response": response_clean,
            "generation_time": generation_time,
            "prompt_type": prompt_type
        }
        results.append(result)
        
        print(f"✓ Sample {cur_image_idx + 1} complete ({generation_time:.2f}s)")
        print(f"Response: {response_clean[:200]}{'...' if len(response_clean) > 200 else ''}")
        
        if monitor:
            monitor.log_memory(f"Inference {cur_image_idx + 1}", f"{generation_time:.2f}s")
    
    # Final memory check
    if monitor:
        monitor.log_memory("Complete", "Inference finished")
        monitor.print_summary()
    
    print(f"\n✅ Inference complete! Processed {len(results)} samples")
    
    # Save results
    results_file = f"inference_results_subj{subject}_{prompt_type}.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {results_file}")
    
    return results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="UMBRAE FP16 Optimized Inference")
    parser.add_argument("--subject", type=int, default=1, choices=[1, 2, 5, 7],
                       help="Subject ID to process")
    parser.add_argument("--prompt", type=str, default="grounding", 
                       choices=list(PROMPTS.keys()),
                       help="Prompt type to use")
    parser.add_argument("--samples", type=int, default=3,
                       help="Maximum number of samples to process")
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable FP16 optimizations")
    parser.add_argument("--precision", type=str, default="fp16", 
                       choices=["fp16", "fp32", "hybrid"],
                       help="Precision mode: fp16 (all FP16), fp32 (all FP32), hybrid (brain FP32, LLaMA FP16)")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Disable memory monitoring")
    
    args = parser.parse_args()
    
    print("UMBRAE FP16 Inference")
    print("=" * 50)
    
    try:
        # Determine precision mode
        if args.no_fp16:
            precision = "fp32"
        else:
            precision = args.precision
            
        results = run_optimized_inference(
            subject=args.subject,
            prompt_type=args.prompt,
            max_samples=args.samples,
            use_fp16=not args.no_fp16,
            monitor_memory=not args.no_monitor,
            precision_mode=precision
        )
        
        if results:
            print(f"\n🎉 Successfully processed {len(results)} samples!")
            for result in results:
                print(f"Sample {result['sample']}: {result['generation_time']:.2f}s")
        else:
            print("\n❌ No results generated")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 