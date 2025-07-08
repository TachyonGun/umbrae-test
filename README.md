# UMBRAE: Unified Multimodal Brain Decoding - Modified Version

This is a modified version of the original [UMBRAE](https://github.com/weihaox/UMBRAE) repository designed for easy setup and cross-compatibility across different computing environments.

## Key Modifications

- **Cross-platform compatibility**: Works on HPCs, Google Colab, and consumer GPUs
- **16GB VRAM target**: Optimized for consumer hardware with memory-efficient implementations
- **Easy setup**: Automated environment configuration for different platforms
- **FP16 inference**: Memory-optimized inference with half-precision support

## Quick Start

### Google Colab
1. Clone this repository: `!git clone https://github.com/your-username/umbrae-test.git`
2. Run the setup cell from `colab_setup.py`
3. Download data and checkpoints as needed

### Local/HPC Setup
1. Use the Cursor background agent configuration in `.cursor/environment.json`
2. Or manually create conda environment: `conda create -n brainx python=3.8`
3. Install dependencies: `pip install -r umbrae/requirements.txt`

### Available Scripts
- `run_inference_fp16.py`: Memory-efficient inference
- `umbrae_inference_fp16.py`: Optimized brain decoding pipeline
- `memory_utils.py`: Memory management utilities
- `config.py`: Configuration management

## Original Paper
```bibtex
@inproceedings{xia2024umbrae,
  author    = {Xia, Weihao and de Charette, Raoul and Öztireli, Cengiz and Xue, Jing-Hao},
  title     = {UMBRAE: Unified Multimodal Brain Decoding},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```

## License
See LICENSE.txt for details.

---
**Note**: This modified version focuses on accessibility and ease of use. For the original research implementation, visit the [official UMBRAE repository](https://github.com/weihaox/UMBRAE).