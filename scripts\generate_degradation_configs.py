"""
Generate 240 compound degradation configuration files.

This script creates JSON configuration files for all degradation
scenarios used in the Visual Observability Benchmark.
"""

import json
import numpy as np
from pathlib import Path


def generate_classical_sr_configs():
    """Generate classical super-resolution configurations."""
    configs = []
    scales = [2, 3, 4, 8]
    
    for scale in scales:
        count = 20 if scale == 4 else 15 if scale in [2, 3] else 10
        for i in range(count):
            configs.append({
                "degradation_id": f"classical_sr_x{scale}_{i:03d}",
                "category": "classical_sr",
                "parameters": {
                    "scale": scale,
                    "interpolation": "bicubic"
                },
                "difficulty": "easy" if scale <= 2 else "medium" if scale == 4 else "hard",
                "description": f"{scale}x bicubic downsampling"
            })
    
    return configs


def generate_gaussian_blur_configs():
    """Generate Gaussian blur configurations."""
    configs = []
    kernel_sizes = [7, 15, 21]
    sigma_ranges = {
        7: np.linspace(0.5, 3.0, 10),
        15: np.linspace(1.0, 5.0, 15),
        21: np.linspace(2.0, 7.0, 15)
    }
    
    for kernel_size in kernel_sizes:
        for sigma in sigma_ranges[kernel_size]:
            configs.append({
                "degradation_id": f"gaussian_blur_k{kernel_size}_s{sigma:.1f}",
                "category": "gaussian_blur",
                "parameters": {
                    "kernel_size": kernel_size,
                    "sigma": float(sigma)
                },
                "difficulty": "easy" if sigma < 2 else "medium" if sigma < 4 else "hard",
                "description": f"Gaussian blur (kernel={kernel_size}, sigma={sigma:.1f})"
            })
    
    return configs


def generate_noise_configs():
    """Generate noise configurations."""
    configs = []
    
    # Gaussian noise
    for sigma in np.linspace(5, 50, 15):
        configs.append({
            "degradation_id": f"noise_gaussian_s{sigma:.0f}",
            "category": "noise",
            "parameters": {
                "type": "gaussian",
                "sigma": float(sigma)
            },
            "difficulty": "easy" if sigma < 15 else "medium" if sigma < 35 else "hard",
            "description": f"Gaussian noise (sigma={sigma:.0f})"
        })
    
    # Poisson noise
    for scale in np.linspace(1, 10, 10):
        configs.append({
            "degradation_id": f"noise_poisson_s{scale:.1f}",
            "category": "noise",
            "parameters": {
                "type": "poisson",
                "scale": float(scale)
            },
            "difficulty": "easy" if scale < 3 else "medium" if scale < 6 else "hard",
            "description": f"Poisson noise (scale={scale:.1f})"
        })
    
    # Speckle noise
    for sigma in np.linspace(0.1, 0.5, 10):
        configs.append({
            "degradation_id": f"noise_speckle_s{sigma:.2f}",
            "category": "noise",
            "parameters": {
                "type": "speckle",
                "sigma": float(sigma)
            },
            "difficulty": "easy" if sigma < 0.2 else "medium" if sigma < 0.35 else "hard",
            "description": f"Speckle noise (sigma={sigma:.2f})"
        })
    
    # Mixed noise
    for i in range(5):
        configs.append({
            "degradation_id": f"noise_mixed_{i:03d}",
            "category": "noise",
            "parameters": {
                "type": "mixed",
                "gaussian_sigma": float(np.random.uniform(10, 30)),
                "poisson_scale": float(np.random.uniform(2, 5)),
                "speckle_sigma": float(np.random.uniform(0.1, 0.3))
            },
            "difficulty": "hard",
            "description": "Mixed noise (Gaussian + Poisson + Speckle)"
        })
    
    return configs


def generate_jpeg_configs():
    """Generate JPEG compression configurations."""
    configs = []
    quality_ranges = [
        (10, 30, 10),
        (31, 60, 10),
        (61, 90, 10)
    ]
    
    for q_min, q_max, count in quality_ranges:
        for quality in np.linspace(q_min, q_max, count, dtype=int):
            configs.append({
                "degradation_id": f"jpeg_q{quality}",
                "category": "jpeg_compression",
                "parameters": {
                    "quality": int(quality)
                },
                "difficulty": "hard" if quality < 30 else "medium" if quality < 60 else "easy",
                "description": f"JPEG compression (quality={quality})"
            })
    
    return configs


def generate_realworld_configs():
    """Generate real-world combined degradation configurations."""
    configs = []
    
    combinations = [
        ("blur_noise", 20),
        ("blur_compression", 15),
        ("noise_compression", 15),
        ("blur_noise_compression", 20)
    ]
    
    for combo_type, count in combinations:
        for i in range(count):
            if combo_type == "blur_noise":
                params = {
                    "blur_kernel": int(np.random.choice([7, 15, 21])),
                    "blur_sigma": float(np.random.uniform(1.0, 5.0)),
                    "noise_type": str(np.random.choice(["gaussian", "poisson"])),
                    "noise_sigma": float(np.random.uniform(10, 30))
                }
            elif combo_type == "blur_compression":
                params = {
                    "blur_kernel": int(np.random.choice([7, 15, 21])),
                    "blur_sigma": float(np.random.uniform(1.0, 5.0)),
                    "jpeg_quality": int(np.random.uniform(20, 80))
                }
            elif combo_type == "noise_compression":
                params = {
                    "noise_type": str(np.random.choice(["gaussian", "poisson", "speckle"])),
                    "noise_sigma": float(np.random.uniform(10, 30)),
                    "jpeg_quality": int(np.random.uniform(20, 80))
                }
            else:  # blur_noise_compression
                params = {
                    "blur_kernel": int(np.random.choice([7, 15, 21])),
                    "blur_sigma": float(np.random.uniform(1.0, 5.0)),
                    "noise_type": str(np.random.choice(["gaussian", "poisson"])),
                    "noise_sigma": float(np.random.uniform(10, 30)),
                    "jpeg_quality": int(np.random.uniform(20, 80))
                }
            
            configs.append({
                "degradation_id": f"realworld_{combo_type}_{i:03d}",
                "category": "realworld",
                "parameters": params,
                "difficulty": "hard",
                "description": f"Real-world degradation: {combo_type.replace('_', ' + ')}"
            })
    
    return configs


def main():
    output_dir = Path("./data/degradation_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    all_configs = {
        "classical_sr": generate_classical_sr_configs(),
        "gaussian_blur": generate_gaussian_blur_configs(),
        "noise": generate_noise_configs(),
        "jpeg_compression": generate_jpeg_configs(),
        "realworld": generate_realworld_configs()
    }
    
    # Save individual category files
    for category, configs in all_configs.items():
        filepath = output_dir / f"{category}.json"
        with open(filepath, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Generated {len(configs)} configurations for {category}")
    
    # Save combined file
    combined = []
    for configs in all_configs.values():
        combined.extend(configs)
    
    with open(output_dir / "all_configs.json", 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nTotal configurations: {len(combined)}")
    print(f"All configs saved to: {output_dir}")


if __name__ == '__main__':
    main()
