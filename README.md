# Visual Observability Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Visual Observability: An Information-Theoretic Framework for Recoverability in Image Restoration"** (Neural Computing and Applications).

## Overview

This repository provides a comprehensive benchmark framework for quantifying image recoverability through **Visual Observability Maps**. The framework addresses the critical gap between perceptual realism and physical truth in modern generative image restoration models.

### Key Features

- **240 Compound Degradation Matrices**: Comprehensive benchmark covering diverse real-world degradation scenarios
- **Multi-Model Consensus Framework**: Integration of SwinIR, NAFNet, and DiffBIR
- **Observability Map Generation**: Information-theoretic quantification of recoverability
- **ODNet**: Lightweight U-Net for direct observability prediction from degraded inputs
- **Physical Warning System**: Objective identification of regions where signal is irreversibly lost

## Project Structure

```
Visual_Observability_Release/
├── core/                       # Core algorithms
│   ├── degradation.py          # Degradation engine (classical & real-world)
│   └── observability_math.py   # Observability map computation
├── models/                     # Model wrappers
│   ├── wrapper_swinir.py       # SwinIR integration
│   ├── wrapper_nafnet.py       # NAFNet integration
│   └── wrapper_diffusion.py    # DiffBIR integration
├── pretrained_models/          # Pre-trained model weights (~418MB)
│   ├── swinir/                 # SwinIR pre-trained weights
│   ├── nafnet/                 # NAFNet pre-trained weights
│   └── diffbir/                # DiffBIR pre-trained weights
├── pipelines/                  # Processing pipelines
│   └── dataset_generator.py    # Benchmark dataset generation
├── data/                       # Data directory
│   ├── sample_images/          # Sample test images
│   ├── degradation_configs/    # 240 degradation configurations
│   └── observability_maps/     # Generated observability maps
├── benchmarks/                 # Benchmark results
│   ├── classical/              # Classical degradation benchmark
│   └── realworld/              # Real-world degradation benchmark
├── configs/                    # Configuration files
├── scripts/                    # Utility scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
│   ├── figures/                # Paper figures
│   └── supplementary/          # Supplementary materials
├── main.py                     # Main entry point
├── config.py                   # Configuration management
└── requirements.txt            # Python dependencies
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Visual_Observability_Benchmark.git
cd Visual_Observability_Benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pre-trained models are included in the repository (~418MB)
# No additional download required
```

## Quick Start

### 1. Generate Observability Maps

```bash
python main.py \
    --gt_folder ./data/sample_images \
    --output ./results \
    --scale 4 \
    --deg_type classical
```

### 2. Generate Benchmark Dataset

```bash
python main.py \
    --gt_folder ./data/sample_images \
    --output ./benchmarks/classical \
    --scale 4 \
    --deg_type classical \
    --num_images 100
```

### 3. Real-world Degradation

```bash
python main.py \
    --gt_folder ./data/sample_images \
    --output ./benchmarks/realworld \
    --deg_type realworld
```

## Core Formula

The observability map is computed as:

```
O = exp(-(α × Error + β × Variance))
```

Where:
- **Error**: MSE between consensus mean and ground truth
- **Variance**: Disagreement across restoration models
- **α, β**: Weighting parameters (default: α=1.0, β=0.5)

## 240 Compound Degradation Matrices

Our benchmark includes 240 carefully designed degradation configurations:

| Category | Count | Description |
|----------|-------|-------------|
| Classical SR | 60 | Bicubic downsampling with various scales |
| Gaussian Blur | 40 | Different kernel sizes and sigma values |
| Noise | 40 | Gaussian, Poisson, and speckle noise |
| JPEG Compression | 30 | Various quality factors |
| Real-world | 70 | Combined degradations (blur + noise + compression) |

See `data/degradation_configs/` for complete specifications.

## Model Support

| Model | Task | Pre-trained Weights |
|-------|------|-------------------|
| SwinIR | Classical/Real-world SR | Included |
| NAFNet | Denoising/Deblurring | Included |
| DiffBIR | Blind Image Restoration | Included |

## Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@article{xu2025visual,
  title={Visual Observability: An Information-Theoretic Framework for Recoverability in Image Restoration},
  author={Xu, Yanxin and Song, Jian and Park, Jiyeon and Shao, Ziling},
  journal={Neural Computing and Applications},
  year={2025},
  publisher={Springer}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SwinIR](https://github.com/JingyunLiang/SwinIR) - Swin Transformer for Image Restoration
- [NAFNet](https://github.com/megvii-research/NAFNet) - Simple Baseline for Image Restoration
- [DiffBIR](https://github.com/XPixelGroup/DiffBIR) - Towards Blind Image Restoration with Generative Diffusion Prior

## Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author.

## Data Availability

The datasets generated and/or analysed during the current study are available in this GitHub repository. The complete compound degradation benchmark (240 matrices), source code, and pre-trained models are included.
