# Installation Guide

## System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.8 or higher
- **CUDA**: 11.3 or higher (for GPU support)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but strongly recommended)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Visual_Observability_Benchmark.git
cd Visual_Observability_Benchmark
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Pre-trained Models

Pre-trained models are automatically downloaded on first use. Alternatively, you can manually download them:

```bash
# Create models directory
mkdir -p pretrained_models/{swinir,nafnet,diffbir}

# Download links will be provided in the repository releases
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
python main.py --device cpu
```

### Model Download Issues

If automatic download fails, manually download from:
- SwinIR: [Official Repository](https://github.com/JingyunLiang/SwinIR)
- NAFNet: [Official Repository](https://github.com/megvii-research/NAFNet)
- DiffBIR: [Official Repository](https://github.com/XPixelGroup/DiffBIR)

## Docker Support (Optional)

```bash
docker build -t vob .
docker run --gpus all -it -v $(pwd)/data:/data vob
```
