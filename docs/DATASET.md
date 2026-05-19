# Dataset Guide

## Overview

The Visual Observability Benchmark includes **240 compound degradation matrices** designed to comprehensively evaluate image recoverability across diverse scenarios.

## Degradation Categories

### 1. Classical Super-Resolution (60 configurations)

| Scale | Configurations |
|-------|---------------|
| x2 | 15 |
| x3 | 15 |
| x4 | 20 |
| x8 | 10 |

**Degradation**: Bicubic downsampling

### 2. Gaussian Blur (40 configurations)

| Kernel Size | Sigma Range | Count |
|-------------|-------------|-------|
| 7x7 | 0.5 - 3.0 | 10 |
| 15x15 | 1.0 - 5.0 | 15 |
| 21x21 | 2.0 - 7.0 | 15 |

### 3. Noise (40 configurations)

| Type | Sigma Range | Count |
|------|-------------|-------|
| Gaussian | 5 - 50 | 15 |
| Poisson | 1 - 10 | 10 |
| Speckle | 0.1 - 0.5 | 10 |
| Mixed | Combined | 5 |

### 4. JPEG Compression (30 configurations)

| Quality Factor | Count |
|----------------|-------|
| 10 - 30 | 10 |
| 31 - 60 | 10 |
| 61 - 90 | 10 |

### 5. Real-world Combined (70 configurations)

Combinations of:
- Blur + Noise
- Blur + Compression
- Noise + Compression
- Blur + Noise + Compression

## Directory Structure

```
data/
├── sample_images/          # Example ground truth images
│   ├── checkerboard.png
│   ├── gradient_blue.png
│   ├── gradient_red.png
│   ├── pattern_stripes.png
│   ├── texture_noise.png
│   └── texture_smooth.png
├── degradation_configs/    # JSON configuration files
│   ├── classical_sr.json
│   ├── gaussian_blur.json
│   ├── noise.json
│   ├── jpeg_compression.json
│   └── realworld_combined.json
└── observability_maps/     # Generated observability maps
```

## Configuration Format

Each degradation configuration is stored as JSON:

```json
{
  "degradation_id": "classical_sr_x4_001",
  "category": "classical_sr",
  "parameters": {
    "scale": 4,
    "interpolation": "bicubic"
  },
  "difficulty": "easy",
  "description": "4x bicubic downsampling"
}
```

## Usage

### Generate Custom Dataset

```python
from pipelines.dataset_generator import DatasetGenerator

generator = DatasetGenerator(
    gt_folder="./data/sample_images",
    output_folder="./output",
    scale=4,
    degradation_type="classical"
)

generator.generate()
```

### Load Degradation Config

```python
import json

with open("data/degradation_configs/classical_sr.json") as f:
    configs = json.load(f)

print(f"Total configurations: {len(configs)}")
```

## Data Format

### Output Structure

Each generated sample includes:

```
output/
├── degraded/           # Degraded image
├── ground_truth/       # Original image
├── restored/          # Restored images from each model
│   ├── *_swinir.png
│   ├── *_nafnet.png
│   └── *_diffbir.png
├── observability/     # Observability map
└── metadata/         # JSON with degradation params and metrics
```

### Metadata Format

```json
{
  "image_name": "checkerboard",
  "degradation": {
    "type": "classical_sr",
    "scale": 4,
    "parameters": {...}
  },
  "observability": {
    "mean": 0.75,
    "std": 0.15,
    "min": 0.20,
    "max": 0.98
  },
  "models_used": ["swinir", "nafnet", "diffbir"],
  "processing_time": 12.5
}
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{xu2025visual,
  title={Visual Observability: An Information-Theoretic Framework for Recoverability in Image Restoration},
  author={Xu, Yanxin and Song, Jian and Park, Jiyeon and Shao, Ziling},
  journal={Neural Computing and Applications},
  year={2025}
}
```
