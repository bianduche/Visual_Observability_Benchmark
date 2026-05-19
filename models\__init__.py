"""
Visual Observability Benchmark - Models Module
================================================
Model wrappers for SOTA image restoration models.

Supports:
- SwinIR: Swin Transformer for Image Restoration
- NAFNet: Simple Attention-free Network
- DiffBIR: Diffusion-based Image Restoration

Author: Visual Observability Benchmark Team
"""

from .wrapper_swinir import SwinIRWrapper
from .wrapper_nafnet import NAFNetWrapper
from .wrapper_diffusion import DiffBIRWrapper

__all__ = ['SwinIRWrapper', 'NAFNetWrapper', 'DiffBIRWrapper']
