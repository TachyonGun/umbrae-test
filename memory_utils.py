"""
Memory monitoring utilities for UMBRAE FP16 inference
Provides detailed memory tracking and optimization helpers
"""

import torch
import gc
import psutil
import time
from typing import Dict, Optional, List
from contextlib import contextmanager

class MemoryMonitor:
    """Monitor and track GPU/CPU memory usage"""
    
    def __init__(self):
        self.memory_log: List[Dict] = []
        self.peak_memory = 0
        self.start_time = time.time()
    
    def log_memory(self, stage: str, details: Optional[str] = None) -> Dict:
        """Log current memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        # GPU memory
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1024**3
        cpu_available = cpu_memory.available / 1024**3
        
        # Track peak
        if gpu_allocated > self.peak_memory:
            self.peak_memory = gpu_allocated
        
        memory_info = {
            "timestamp": time.time() - self.start_time,
            "stage": stage,
            "details": details or "",
            "gpu_allocated_gb": round(gpu_allocated, 2),
            "gpu_reserved_gb": round(gpu_reserved, 2), 
            "gpu_free_gb": round(gpu_free, 2),
            "cpu_used_gb": round(cpu_used, 2),
            "cpu_available_gb": round(cpu_available, 2),
            "peak_gpu_gb": round(self.peak_memory, 2)
        }
        
        self.memory_log.append(memory_info)
        
        # Print memory info with color coding
        if gpu_allocated > 12:
            color = "\033[91m"  # Red
        elif gpu_allocated > 8:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[92m"  # Green
        
        reset_color = '\033[0m'
        print(f"{color}[{stage}]{reset_color} GPU: {gpu_allocated:.2f}GB allocated, "
              f"{gpu_reserved:.2f}GB reserved, {gpu_free:.2f}GB free")
        
        if details:
            print(f"  └─ {details}")
        
        return memory_info
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory usage"""
        if not self.memory_log:
            return {}
        
        return {
            "peak_gpu_memory_gb": self.peak_memory,
            "final_gpu_memory_gb": self.memory_log[-1]["gpu_allocated_gb"],
            "total_stages": len(self.memory_log),
            "duration_seconds": self.memory_log[-1]["timestamp"]
        }
    
    def print_summary(self):
        """Print memory usage summary"""
        summary = self.get_memory_summary()
        if not summary:
            print("No memory data collected")
            return
        
        print("\n" + "="*50)
        print("MEMORY USAGE SUMMARY")
        print("="*50)
        print(f"Peak GPU Memory: {summary['peak_gpu_memory_gb']:.2f}GB")
        print(f"Final GPU Memory: {summary['final_gpu_memory_gb']:.2f}GB")
        print(f"Total Duration: {summary['duration_seconds']:.1f}s")
        print(f"Monitoring Points: {summary['total_stages']}")
        
        # Show progression
        print("\nMemory progression:")
        for entry in self.memory_log:
            print(f"  {entry['timestamp']:6.1f}s - {entry['stage']:20s} - "
                  f"{entry['gpu_allocated_gb']:5.2f}GB")

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

@contextmanager
def memory_efficient_loading(enable_fp16: bool = True):
    """Context manager for memory-efficient model loading"""
    original_dtype = torch.get_default_dtype()
    
    try:
        if enable_fp16:
            torch.set_default_dtype(torch.float16)
        
        cleanup_gpu_memory()
        yield
        
    finally:
        torch.set_default_dtype(original_dtype)
        cleanup_gpu_memory()

def check_memory_requirements(required_gb: float = 9.0) -> bool:
    """Check if system has enough GPU memory"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available_memory = total_memory - (torch.cuda.memory_allocated() / 1024**3)
    
    print(f"GPU Memory Check:")
    print(f"  Total: {total_memory:.1f}GB")
    print(f"  Available: {available_memory:.1f}GB") 
    print(f"  Required: {required_gb:.1f}GB")
    
    if available_memory >= required_gb:
        print(f"✓ Sufficient memory available")
        return True
    else:
        print(f"❌ Insufficient memory. Need {required_gb - available_memory:.1f}GB more")
        return False

def optimize_model_for_inference(model, enable_fp16: bool = True):
    """Apply inference optimizations to a model"""
    print("Applying inference optimizations...")
    
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Convert to FP16 if requested
    if enable_fp16:
        model.half()
        print("✓ Model converted to FP16")
    
    # Enable optimizations
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True
    
    if hasattr(torch.backends.cudnn, 'enabled'):
        torch.backends.cudnn.enabled = True
    
    print("✓ Inference optimizations applied")
    return model

def get_optimal_batch_size(model_size_gb: float, available_memory_gb: float) -> int:
    """Calculate optimal batch size based on available memory"""
    # Reserve memory for activations and other overhead
    usable_memory = available_memory_gb * 0.8  # Use 80% of available memory
    
    # Estimate memory per sample (rough heuristic)
    memory_per_sample = model_size_gb * 0.1  # 10% of model size per sample
    
    optimal_batch = max(1, int(usable_memory / memory_per_sample))
    
    print(f"Memory analysis:")
    print(f"  Available: {available_memory_gb:.1f}GB")
    print(f"  Usable: {usable_memory:.1f}GB")  
    print(f"  Per sample: {memory_per_sample:.1f}GB")
    print(f"  Optimal batch size: {optimal_batch}")
    
    return min(optimal_batch, 4)  # Cap at 4 for stability 