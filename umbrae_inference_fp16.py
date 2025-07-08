#!/usr/bin/env python3
"""
UMBRAE Inference Script with FP16 Optimizations
Optimized for 16GB VRAM machines

This script replicates the playground notebook functionality with memory optimizations.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import torch
import time
import json

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("CUDA not available")

def check_system_requirements():
    """Check system requirements similar to notebook cells 1-2"""
    print("=== System Requirements Check ===")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU Memory: {total_memory:.1f}GB")
        
        if total_memory < 15:
            print("⚠️  WARNING: Less than 16GB GPU memory detected. Performance may be limited.")
    else:
        print("❌ CUDA not available")
        return False
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    print_gpu_memory()
    return True

def setup_repositories():
    """Clone repositories if they don't exist"""
    print("\n=== Repository Setup ===")
    
    repos = [
        ("https://github.com/weihaox/UMBRAE.git", "UMBRAE"),
        ("https://github.com/weihaox/BrainHub.git", "BrainHub")
    ]
    
    for repo_url, repo_name in repos:
        if not os.path.exists(repo_name):
            print(f"Cloning {repo_name}...")
            subprocess.run(["git", "clone", repo_url], check=True)
            print(f"✓ {repo_name} cloned successfully")
        else:
            print(f"✓ {repo_name} already exists")

def install_dependencies():
    """Install required packages"""
    print("\n=== Installing Dependencies ===")
    
    required_packages = [
        "torch", "torchvision", "transformers", "accelerate==0.19.0",
        "sentencepiece", "braceexpand", "webdataset", "einops", "einops_exts",
        "huggingface_hub", "numpy", "pillow", "tqdm"
    ]
    
    for package in required_packages:
        try:
            __import__(package.split('==')[0].replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✓ {package} installed")

def download_test_data():
    """Download test datasets"""
    print("\n=== Downloading Test Data ===")
    
    data_dir = Path("UMBRAE/umbrae/nsd_data/webdataset_avg_split/test")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = [1, 2, 5, 7]
    base_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/test"
    
    for subj in subjects:
        for idx in [0, 1]:
            filename = f"test_subj0{subj}_{idx}.tar"
            filepath = data_dir / filename
            
            if filepath.exists():
                print(f"✓ {filename} already exists")
                continue
                
            url = f"{base_url}/{filename}"
            print(f"Downloading {filename}...")
            
            import urllib.request
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ {filename} downloaded")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

def download_reference_images():
    """Download reference images for BrainHub"""
    print("\n=== Downloading Reference Images ===")
    
    caption_dir = Path("BrainHub/caption")
    caption_dir.mkdir(parents=True, exist_ok=True)
    
    images_file = caption_dir / "all_images.pt"
    if not images_file.exists():
        url = "https://huggingface.co/datasets/weihaox/brainx/resolve/main/all_images.pt"
        print("Downloading reference images...")
        
        import urllib.request
        urllib.request.urlretrieve(url, images_file)
        print("✓ Reference images downloaded")
    else:
        print("✓ Reference images already exist")
    
    # Process images
    if os.path.exists("BrainHub/processing/decode_images.py"):
        print("Processing reference images...")
        subprocess.run([sys.executable, "processing/decode_images.py"], 
                      cwd="BrainHub", check=True)
        print("✓ Reference images processed")

def download_checkpoints():
    """Download model checkpoints"""
    print("\n=== Downloading Model Checkpoints ===")
    
    from huggingface_hub import snapshot_download
    
    checkpoint_dir = Path("UMBRAE/umbrae")
    if not (checkpoint_dir / "train_logs").exists():
        print("Downloading checkpoints from Hugging Face...")
        snapshot_download(
            repo_id="weihaox/brainx", 
            repo_type="dataset", 
            local_dir=str(checkpoint_dir),
            ignore_patterns=["all_images.pt", ".gitattributes"]
        )
        print("✓ Checkpoints downloaded")
    else:
        print("✓ Checkpoints already exist")

def optimize_models_fp16():
    """Apply FP16 optimizations to models"""
    print("\n=== Applying FP16 Optimizations ===")
    
    # Modify model.py to enable FP16 optimizations
    model_file = Path("UMBRAE/umbrae/model.py")
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Enable CLIP half precision (uncomment the line)
    if "# param.data = param.data.half()" in content:
        content = content.replace(
            "# param.data = param.data.half()", 
            "param.data = param.data.half()  # FP16 optimization enabled"
        )
        print("✓ Enabled CLIP FP16 optimization in model.py")
        
        with open(model_file, 'w') as f:
            f.write(content)
    else:
        print("✓ CLIP FP16 optimization already enabled or not found")

def create_optimized_inference_script():
    """Create an optimized inference script"""
    print("\n=== Creating Optimized Inference Script ===")
    
    inference_script = """
import os
import sys
import torch
import time
from pathlib import Path

# Add UMBRAE to path
sys.path.append('UMBRAE/umbrae')

from transformers import LlamaForCausalLM, LlamaTokenizer
import braceexpand
import webdataset as wds
from model import BrainX
import utils

def print_memory_usage(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def run_optimized_inference():
    print("=== Starting Optimized Inference ===")
    
    # Configuration
    data_path = "UMBRAE/umbrae/nsd_data"
    subj = 1
    prompt = "Please interpret this image and give coordinates [x1,y1,x2,y2] for each object you mention."
    brainx_path = "UMBRAE/umbrae/train_logs/brainx.pth"
    shikra_path = "UMBRAE/umbrae/model_weights/shikra-7b"
    adapter_path = "UMBRAE/umbrae/model_weights/mm_projector.bin"
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print_memory_usage("Initial")
    
    # Load data
    print("Loading test data...")
    val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
    val_batch_size = 1
    
    val_url = list(braceexpand.braceexpand(val_url))
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=data_path) \\
        .decode("torch")\\
        .rename(images="jpg;png", voxels='nsdgeneral.npy', trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy") \\
        .to_tuple("voxels", "images") \\
        .batched(val_batch_size, partial=False)
    
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, num_workers=1, shuffle=False)
    
    # Load brain encoder with FP16
    print("Loading brain encoder...")
    voxel2emb = BrainX(hidden_dim=1024, out_dim=1024, num_latents=256)
    voxel2emb.to(device)
    
    checkpoint = torch.load(brainx_path, map_location='cpu')
    voxel2emb.load_state_dict(checkpoint['model_state_dict'], strict=False)
    voxel2emb.eval()
    
    # Convert brain encoder to FP16
    print("✓ Converting brain encoder to FP16...")
    voxel2emb.half()
    
    print_memory_usage("After Brain Encoder")
    
    # Process brain signals
    print("Processing brain signals...")
    emb_voxel_list = []
    for val_i, (voxel, image) in enumerate(val_dl):
        if val_i >= 3:  # Process only first 3 samples for demo
            break
            
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                voxel = torch.mean(voxel, axis=1).half()  # Convert to FP16
                emb_voxel = voxel2emb(voxel.to(device), modal=f'fmri{subj}')
                emb_voxel_list.append(emb_voxel)
    
    image_features = torch.cat(emb_voxel_list, dim=0)
    print(f"✓ Processed {len(emb_voxel_list)} brain signals")
    print_memory_usage("After Brain Processing")
    
    # Load LLaMA model with FP16
    print("Loading LLaMA model with FP16...")
    tokenizer = LlamaTokenizer.from_pretrained(shikra_path, padding_side='left')
    
    # Load model in FP16 to save memory
    model = LlamaForCausalLM.from_pretrained(
        shikra_path, 
        torch_dtype=torch.float16,  # FP16 optimization
        device_map="auto"
    )
    print("✓ LLaMA model loaded in FP16")
    
    print_memory_usage("After LLaMA Model")
    
    # Load MM projector and convert to FP16
    print("Loading MM projector...")
    mm_projector = torch.nn.Linear(1024, 4096)
    mm_projector_weights = torch.load(adapter_path, map_location='cpu')
    
    if adapter_path.endswith('mm_projector.bin'):
        adjusted_state_dict = {k.split('.')[-1]: v for k, v in mm_projector_weights.items()}
        mm_projector.load_state_dict(adjusted_state_dict)
    else:
        mm_projector.load_state_dict(mm_projector_weights['model_state_dict'], strict=False)
    
    mm_projector.to(device).half()  # FP16 optimization
    print("✓ MM projector loaded in FP16")
    
    # Project features
    image_features = mm_projector(image_features.half())
    print(f"✓ Image features shape: {image_features.shape}")
    
    print_memory_usage("After MM Projector")
    
    # Prepare prompt
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    user_image = " <im_start>" + "<im_patch>" * 256 + "<im_end> "
    
    if '<image>' in prompt:
        user_prompt = prompt.replace('<image>', user_image)
    else:
        user_prompt = prompt + user_image
    input_text = system + user_prompt + " ASSISTANT:"
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    # Generation parameters
    gen_kwargs = dict(
        use_cache=True,
        do_sample=False,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        max_new_tokens=512,
    )
    
    # Run inference on first sample
    print("Running inference...")
    cur_image_features = image_features[0].to(device=inputs_embeds.device)
    num_patches = cur_image_features.shape[0]
    
    # Find image start tokens
    image_start_tokens = torch.where(input_ids == 32001)[0]
    if len(image_start_tokens) > 0:
        image_start_token_pos = image_start_tokens[0]
        
        # Insert image features
        new_input_embeds = torch.cat((
            inputs_embeds[:, :image_start_token_pos + 1], 
            cur_image_features.unsqueeze(0),
            inputs_embeds[:, image_start_token_pos + num_patches + 1:]
        ), dim=1)
        
        # Generate response
        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = model.generate(inputs_embeds=new_input_embeds, **gen_kwargs)
        
        generation_time = time.time() - start_time
        response = tokenizer.batch_decode(output_ids)[0]
        
        print(f"\\n=== Generation Complete ({generation_time:.2f}s) ===")
        print(f"Response: {response.strip(' <s></s>')}")
        
        print_memory_usage("Final")
        
        return response
    else:
        print("❌ No image tokens found in prompt")
        return None

if __name__ == "__main__":
    result = run_optimized_inference()
    print("\\n=== Inference Complete ===")
"""
    
    with open("optimized_inference.py", "w") as f:
        f.write(inference_script)
    
    print("✓ Optimized inference script created: optimized_inference.py")

def main():
    """Main execution function"""
    print("UMBRAE FP16 Optimization Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Setup
    setup_repositories()
    install_dependencies()
    
    # Download data and models
    download_test_data()
    download_reference_images()
    download_checkpoints()
    
    # Apply optimizations
    optimize_models_fp16()
    create_optimized_inference_script()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete! You can now run:")
    print("   python optimized_inference.py")
    print("\nExpected memory usage with FP16 optimizations:")
    print("   - Brain Encoder: ~1.0GB")
    print("   - CLIP Model: ~0.9GB") 
    print("   - LLaMA-7B: ~7.0GB")
    print("   - MM Projector: ~0.002GB")
    print("   - Total: ~9GB (fits in 16GB with headroom)")

if __name__ == "__main__":
    main() 