# UMBRAE FP16 Optimization for 16GB GPUs

This repository contains FP16-optimized scripts for running UMBRAE inference on 16GB GPU machines. The optimizations reduce memory usage from ~20GB to ~9GB while maintaining model performance.

## 🚀 Quick Start

### 1. Setup (First Time Only)
```bash
# Run the setup script to download models and data
python umbrae_inference_fp16.py
```

This will:
- ✅ Check system requirements (GPU, CUDA, memory)
- 📦 Clone UMBRAE and BrainHub repositories  
- 🔧 Install required dependencies
- 📥 Download test datasets (subjects 1, 2, 5, 7)
- 🖼️ Download reference images
- 💾 Download model checkpoints (~1.76GB)
- ⚡ Apply FP16 optimizations to model files

### 2. Run Inference
```bash
# Basic inference with default settings
python run_inference_fp16.py

# With custom options
python run_inference_fp16.py --subject 2 --prompt caption --samples 5

# Disable FP16 (if you have >20GB VRAM)
python run_inference_fp16.py --no-fp16
```

## 🔧 Configuration

### Command Line Options
- `--subject {1,2,5,7}`: Subject ID to process (default: 1)
- `--prompt {grounding,caption,qa,custom}`: Prompt type (default: grounding)
- `--samples N`: Max samples to process (default: 3)
- `--no-fp16`: Disable FP16 optimizations 
- `--no-monitor`: Disable memory monitoring

### Configuration Files

**`config.py`** - Modify model paths, inference settings, and memory limits
```python
# Adjust for your GPU memory
MEMORY_CONFIG = {
    "max_memory_gb": 14,  # Leave 2GB headroom
    "enable_cpu_offload": False,
}

# Change prompts
PROMPTS = {
    "grounding": "Please interpret this image and give coordinates [x1,y1,x2,y2] for each object you mention.",
    "caption": "Describe this image <image> as simply as possible.",
    # Add custom prompts...
}
```

## 💾 Memory Optimizations Applied

### FP16 Conversions
- **Brain Encoder**: FP32 → FP16 (~2.0GB → ~1.0GB)
- **CLIP Model**: FP32 → FP16 (~1.7GB → ~0.9GB)  
- **LLaMA-7B**: FP32 → FP16 (~14GB → ~7.0GB)
- **MM Projector**: FP32 → FP16 (~4MB → ~2MB)

### Memory Management
- 🧹 Aggressive garbage collection
- 📊 Real-time memory monitoring
- 🔄 Model offloading (brain encoder cleared before LLaMA)
- ⚡ Mixed precision inference
- 💨 Gradient checkpointing

### Expected Memory Usage
```
Component          FP32    FP16    Saved
Brain Encoder      2.0GB   1.0GB   1.0GB
CLIP Model         1.7GB   0.9GB   0.8GB  
LLaMA-7B          14.0GB   7.0GB   7.0GB
MM Projector       0.004GB 0.002GB 0.002GB
Activations        2.0GB   1.0GB   1.0GB
Total             19.7GB   9.9GB   9.8GB
```

## 📊 Memory Monitoring

The scripts include detailed memory monitoring:

```bash
[Startup] GPU: 0.12GB allocated, 0.50GB reserved, 15.50GB free
[Brain Encoder] GPU: 1.23GB allocated, 2.00GB reserved, 14.00GB free
  └─ FP16: True
[LLaMA Model] GPU: 8.45GB allocated, 9.00GB reserved, 7.00GB free
  └─ Memory limit: 14GB
[Complete] GPU: 9.12GB allocated, 10.00GB reserved, 6.00GB free

MEMORY USAGE SUMMARY
Peak GPU Memory: 9.45GB
Final GPU Memory: 9.12GB
Total Duration: 45.2s
```

## 🔍 Verification Prints

Key optimization verification points:

```python
# FP16 Model Loading
print("✓ Converting brain encoder to FP16...")
print("✓ LLaMA model loaded in FP16")  
print("✓ MM projector loaded in FP16")

# Memory Management
print("✓ Inference optimizations applied")
print("✓ Brain encoder cleared from memory")
print("✓ GPU memory cleaned up")
```

## 📂 File Structure

```
├── umbrae_inference_fp16.py    # Setup script (run first)
├── run_inference_fp16.py       # Main inference script  
├── config.py                   # Configuration settings
├── memory_utils.py             # Memory monitoring utilities
├── README_FP16.md             # This file
└── UMBRAE/                     # Downloaded repository
    └── umbrae/
        ├── model.py            # Modified for FP16
        ├── train_logs/         # Model checkpoints
        └── nsd_data/          # Test datasets
```

## 🐛 Troubleshooting

### Out of Memory Errors
1. **Reduce sample count**: `--samples 1`
2. **Lower memory limit** in `config.py`: `"max_memory_gb": 12`
3. **Enable CPU offload**: `"enable_cpu_offload": True`
4. **Check background processes**: Close other GPU applications

### Import Errors
```bash
# Reinstall dependencies
pip install torch torchvision transformers accelerate sentencepiece
```

### Missing Data/Checkpoints
```bash
# Re-run setup
python umbrae_inference_fp16.py
```

### Performance Issues
- Ensure FP16 is enabled: Remove `--no-fp16` flag
- Check GPU utilization: `nvidia-smi`
- Verify CUDA version compatibility

## 📈 Performance Benchmarks

**16GB RTX 4090** (example):
- Memory usage: ~9-10GB peak
- Inference time: ~15-25s per sample
- Throughput: ~2-4 samples/minute

**12GB RTX 3060** (minimum):
- Memory usage: ~11-12GB peak  
- Inference time: ~25-35s per sample
- May require `max_memory_gb: 10`

## 🎯 Tips for Best Results

1. **Single sample processing**: Use `--samples 1` for maximum memory efficiency
2. **Monitor memory**: Keep `--no-monitor` off to track usage
3. **Close other apps**: Ensure maximum GPU memory availability
4. **Use appropriate subjects**: Subjects 1,2,5,7 have optimized data
5. **Try different prompts**: Each prompt type produces different results

## 🔗 Original Repository

Based on: [UMBRAE: Unified Multimodal Brain Decoding](https://github.com/weihaox/UMBRAE)

For more details on the original implementation, refer to the main repository and paper. 