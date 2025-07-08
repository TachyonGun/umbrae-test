# UMBRAE Cursor Background Agent Setup

This directory contains the configuration for Cursor Background Agents to work with the UMBRAE project.

## Environment Configuration

The `environment.json` file sets up:

### 🔧 **Installation Process**
- Creates a conda environment `brainx` with Python 3.8
- Installs **CPU-only PyTorch** (suitable for background agents without GPU access)
- Installs all dependencies from `requirements.txt`
- Adds additional packages needed for brain decoding research

### 🚀 **Available Terminals**

1. **UMBRAE Development Terminal**
   - Main development environment
   - Shows available commands for inference, training, and data management
   
2. **Jupyter Server Terminal**
   - Automatically starts a Jupyter notebook server
   - Accessible at `http://localhost:8888`
   - Includes the UMBRAE kernel for interactive development

### 📊 **Key Modifications for Background Agents**

- **GPU PyTorch → CPU PyTorch**: Background agents don't have GPU access, so PyTorch is installed with CPU support only
- **conda activation**: Uses proper conda shell hooks for reliable environment activation
- **Executable permissions**: Sets download scripts as executable

### 💡 **Usage Notes**

- **Data Download**: Use `bash download_data.sh` if you need the NSD dataset
- **Model Checkpoints**: Use `bash download_checkpoint.sh` for pretrained models  
- **Development**: The playground.ipynb notebook is available for interactive exploration
- **Training**: CPU-only training will be slower but allows feature development and testing

### 🔍 **Troubleshooting**

If conda activation fails, the background agent environment may not have conda installed. The setup will attempt to use the base Python environment with pip as a fallback.

For full functionality with large datasets and training, consider using a local environment with GPU support. 