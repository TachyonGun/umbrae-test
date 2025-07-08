# UMBRAE Colab Environment Setup
# Paste this cell after cloning the umbrae-test repository in Google Colab

import os
import subprocess
import sys
import torch

def run_command(command, description=""):
    """Run a command and print status"""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if gpu_info.returncode == 0:
            print("🎮 GPU detected!")
            print(gpu_info.stdout)
            return True
        else:
            print("💻 No GPU detected, will use CPU")
            return False
    except FileNotFoundError:
        print("💻 No NVIDIA drivers found, will use CPU")
        return False

# Check system and GPU
print("🚀 Setting up UMBRAE environment in Google Colab")
print("=" * 50)

# Check if we're in the right directory
if not os.path.exists("umbrae"):
    print("📁 Changing to umbrae directory...")
    if os.path.exists("UMBRAE-main"):
        os.chdir("UMBRAE-main")
    elif os.path.exists("umbrae-test"):
        os.chdir("umbrae-test")
    else:
        print("❌ UMBRAE directory not found. Make sure you've cloned the repository first!")
        print("Run: !git clone https://github.com/weihaox/UMBRAE.git")
        sys.exit(1)

# Move to umbrae subdirectory if it exists
if os.path.exists("umbrae"):
    os.chdir("umbrae")
    print(f"📂 Current directory: {os.getcwd()}")

# Check GPU availability
has_gpu = check_gpu()

# Install PyTorch with appropriate backend
print("\n🔧 Installing PyTorch...")
if has_gpu:
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    print("Installing GPU version of PyTorch...")
else:
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    print("Installing CPU version of PyTorch...")

if not run_command(pytorch_cmd, "Installing PyTorch"):
    print("❌ Failed to install PyTorch")
    sys.exit(1)

# Install requirements
print("\n📦 Installing Python dependencies...")
if os.path.exists("requirements.txt"):
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠️ Some requirements may have failed to install")
else:
    # Install individual packages if requirements.txt not found
    packages = [
        "transformers", "datasets", "pillow", "numpy", "nltk", "pandas",
        "mmengine", "tensorboard", "einops", "tqdm", "matplotlib",
        "accelerate==0.19.0", "SentencePiece", "gradio", "fastapi",
        "uvicorn", "bitsandbytes", "braceexpand", "webdataset",
        "einops_exts", "kornia", "umap-learn", "pycocoevalcap",
        "wcwidth", "huggingface_hub"
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installing {package}")

# Install additional packages
print("\n🔬 Installing additional research packages...")
additional_packages = ["fvcore", "iopath", "pyembree", "cython"]
for package in additional_packages:
    run_command(f"pip install {package}", f"Installing {package}")

# Make scripts executable
print("\n🔑 Setting up download scripts...")
if os.path.exists("download_data.sh"):
    run_command("chmod +x download_data.sh", "Making download_data.sh executable")
if os.path.exists("download_checkpoint.sh"):
    run_command("chmod +x download_checkpoint.sh", "Making download_checkpoint.sh executable")

# Verify installation
print("\n✅ Verifying installation...")
try:
    import torch
    import transformers
    import datasets
    import PIL
    import pandas as pd
    import numpy as np
    
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🤗 Transformers version: {transformers.__version__}")
    print(f"📊 Datasets version: {datasets.__version__}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Running on CPU")
        
except ImportError as e:
    print(f"⚠️ Import error: {e}")

print("\n" + "=" * 50)
print("🎉 UMBRAE environment setup complete!")
print("\n📋 Available commands:")
print("  • python inference.py --help          (run inference)")
print("  • python train.py --help              (train models)")
print("  • bash download_data.sh                (download NSD dataset)")
print("  • bash download_checkpoint.sh          (download pretrained models)")
print("  • python playground.ipynb             (interactive exploration)")

print("\n💡 Quick start:")
print("  1. Download checkpoints: !bash download_checkpoint.sh")
print("  2. Download data: !bash download_data.sh")
print("  3. Run inference: !python inference.py [options]")

print("\n🔍 Current directory contents:")
run_command("ls -la", "")

# Optional: Test basic imports
print("\n🧪 Testing basic functionality...")
try:
    # Test if we can import key modules
    exec("""
import sys
sys.path.append('.')
import utils
print("✅ Utils module imported successfully")
""")
except:
    print("⚠️ Could not import utils module (normal if not in umbrae directory)")

print("\n🌟 Setup completed! You're ready to explore UMBRAE.") 