"""
Generate sample test images for the benchmark.

This script creates synthetic test images with known properties
to validate the observability framework.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def generate_checkerboard(size=512, block_size=32):
    """Generate checkerboard pattern."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = 255
    return Image.fromarray(img)


def generate_gradient(size=512, direction='horizontal', color='blue'):
    """Generate gradient pattern."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if direction == 'horizontal':
        gradient = np.linspace(0, 255, size)
        img[:, :] = gradient[:, np.newaxis, np.newaxis]
    else:
        gradient = np.linspace(0, 255, size)
        img[:, :] = gradient[np.newaxis, :, np.newaxis]
    
    if color == 'red':
        img = img[:, :, 0:1].repeat(3, axis=2)
        img[:, :, 1] = 0
        img[:, :, 2] = 0
    elif color == 'blue':
        img[:, :, 0] = 0
        img[:, :, 1] = 0
        img = img[:, :, 2:3].repeat(3, axis=2)
    
    return Image.fromarray(img)


def generate_stripes(size=512, stripe_width=8):
    """Generate stripe pattern."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, stripe_width * 2):
        img[:, i:i+stripe_width] = 255
    return Image.fromarray(img)


def generate_texture_noise(size=512):
    """Generate noise texture."""
    img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(img)


def generate_texture_smooth(size=512):
    """Generate smooth texture."""
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    Z = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
    img = np.stack([Z, Z, Z], axis=2)
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser(description='Generate sample test images')
    parser.add_argument('--output', type=str, default='./data/sample_images',
                        help='Output directory')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test images
    images = {
        'checkerboard.png': generate_checkerboard(args.size),
        'gradient_blue.png': generate_gradient(args.size, 'horizontal', 'blue'),
        'gradient_red.png': generate_gradient(args.size, 'horizontal', 'red'),
        'pattern_stripes.png': generate_stripes(args.size),
        'texture_noise.png': generate_texture_noise(args.size),
        'texture_smooth.png': generate_texture_smooth(args.size),
    }
    
    for filename, img in images.items():
        img.save(output_dir / filename)
        print(f"Generated: {filename}")
    
    print(f"\nAll sample images saved to: {output_dir}")


if __name__ == '__main__':
    main()
