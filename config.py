# Configuration file for UMBRAE FP16 Inference

# Model paths and settings
MODEL_CONFIG = {
    "shikra_path": "umbrae/umbrae/model_weights/shikra-7b",
    "brainx_path": "umbrae/umbrae/train_logs/brainx/last.pth", 
    "adapter_path": "umbrae/umbrae/model_weights/mm_projector.bin",
    "data_path": "umbrae/umbrae/nsd_data"
}

# Inference settings
INFERENCE_CONFIG = {
    "subject": 1,  # Subject ID (1, 2, 5, 7)
    "batch_size": 1,  # Keep at 1 for memory efficiency
    "max_samples": 3,  # Number of samples to process (set to None for all)
    "max_new_tokens": 512,
    "use_fp16": True,  # Enable FP16 optimizations
    "use_mixed_precision": True,  # Enable mixed precision training
    "precision_mode": "fp16"  # Options: "fp16", "fp32", "hybrid" (brain=FP32, llama=FP16)
}

# Memory optimization settings
MEMORY_CONFIG = {
    "enable_cpu_offload": False,  # Offload some components to CPU if needed
    "gradient_checkpointing": True,  # Save memory during generation
    "max_memory_gb": 14,  # Maximum GPU memory to use (leave 2GB headroom)
    "enable_attention_slicing": True  # Reduce memory usage in attention layers
}

# Prompt templates
PROMPTS = {
    "caption": "Describe this image <image> as simply as possible.",
    "grounding": "Please interpret this image and give coordinates [x1,y1,x2,y2] for each object you mention.",
    "qa": "What do you see in this image <image>? Please provide a detailed description.",
    "custom": "Please interpret this image and give coordinates [x1,y1,x2,y2] for each object you mention."
}

# Download URLs (backup in case primary sources fail)
DOWNLOAD_URLS = {
    "nsd_base": "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split",
    "brainx_checkpoints": "https://huggingface.co/datasets/weihaox/brainx",
    "reference_images": "https://huggingface.co/datasets/weihaox/brainx/resolve/main/all_images.pt"
}

# System requirements
SYSTEM_REQUIREMENTS = {
    "min_gpu_memory_gb": 12,  # Minimum GPU memory required
    "recommended_gpu_memory_gb": 16,
    "min_python_version": (3, 8),
    "required_cuda_version": "11.0"
} 