"""
SwinIR Model Wrapper
====================
Wrapper interface for SwinIR (Swin Transformer for Image Restoration).

SwinIR is a state-of-the-art image restoration model that uses
Swin Transformer architecture for various tasks:
- Classical image super-resolution
- Lightweight image super-resolution
- Real image super-resolution
- Grayscale denoising
- Color image denoising
- JPEG artifact removal

Reference: Liang et al., "SwinIR: Image Restoration Using Swin Transformer"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Tuple
import logging
from pathlib import Path
import sys
import os
import re
import types

logger = logging.getLogger(__name__)


def _parse_swinir_classical_checkpoint(path: Path) -> Tuple[int, int]:
    """
    Infer (upscale, training_patch_size) from official SwinIR checkpoint filenames, e.g.
    001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth -> (8, 48)
    """
    name = path.name
    um = re.search(r"_x(\d+)(?:\.pth|\.pt)$", name, re.IGNORECASE)
    upscale = int(um.group(1)) if um else 4
    if "s48" in name.lower():
        tps = 48
    elif "s64" in name.lower():
        tps = 64
    else:
        tps = 48
    return upscale, tps


class SwinIRWrapper:
    """
    Wrapper class for SwinIR model.

    Provides a clean interface for running SwinIR inference with
    automatic model loading and memory management.

    Args:
        model_path: Path to pretrained SwinIR weights (.pth file)
        device: Computation device ('cuda' or 'cpu')
        model_type: Type of SwinIR model
            - 'classical': Classical image SR (×2, ×3, ×4)
            - 'lightweight': Lightweight classical SR
            - 'real_sr': Real-world image SR (no reference)
            - 'denoise_grayscale': Grayscale image denoising
            - 'denoise_color': Color image denoising
            - 'jpeg': JPEG artifact removal
        scale: Upscaling factor for SR models (1, 2, 3, 4)
        tile_size: Tile size for tile-based inference (0 for full image)
        tile_pad: Padding around each tile
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_type: str = 'classical',
        scale: int = 4,
        tile_size: int = 0,
        tile_pad: int = 16,
        training_patch_size: Optional[int] = None,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.training_patch_size = training_patch_size

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self._loaded = False

    def _get_default_model_path(self) -> Optional[Path]:
        """Get default model path if not specified."""
        if self.model_path:
            return Path(self.model_path)

        # Resolve base once (e:\制作数据集系统\)
        base_path = Path(__file__).resolve().parent.parent.parent

        # Known file locations
        swinir_files = [
            # In models/ folder (project root)
            base_path / 'models' / 'swinir' / '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
            base_path / 'models' / 'swinir' / '001_classicalSR_DIV2K.pth',
            # In SwinIR-main/SwinIR-main/experiments/pretrained_models/
            base_path / 'SwinIR-main' / 'SwinIR-main' / 'experiments' / 'pretrained_models' / '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
            base_path / 'SwinIR-main' / 'SwinIR-main' / 'experiments' / 'pretrained_models' / '001_classicalSR_DIV2K.pth',
            # In project root (some user may have put it here)
            base_path / '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
            base_path / '001_classicalSR_DIV2K.pth',
            # Legacy/alternative locations
            base_path / 'SwinIR-main' / 'SwinIR-main' / '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
        ]

        for p in swinir_files:
            if p.is_file():
                logger.info(f"Found SwinIR weights: {p}")
                return p

        return None

    def _load_model(self) -> None:
        """Load SwinIR model architecture and weights."""
        if self._loaded:
            return

        try:
            from models.network_swinir import SwinIR as net
        except ImportError:
            import importlib.util
            base_path = Path(__file__).resolve().parent.parent.parent

            # SwinIR code: e:\制作数据集系统\SwinIR-main\SwinIR-main\models\network_swinir.py
            candidates = [
                base_path / 'SwinIR-main' / 'SwinIR-main' / 'models' / 'network_swinir.py',
                base_path / 'SwinIR-main' / 'models' / 'network_swinir.py',
            ]
            spec = None
            for model_file in candidates:
                if model_file.is_file():
                    # Try spec_from_file_location (avoids sys.path and module cache)
                    spec = importlib.util.spec_from_file_location(
                        'models.network_swinir', str(model_file)
                    )
                    if spec and spec.loader:
                        break
                    spec = None

            if spec is None:
                raise ImportError(
                    f"SwinIR network_swinir.py not found. Tried: {[str(c) for c in candidates]}"
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules['models.network_swinir'] = module
            sys.modules['models'] = types.SimpleNamespace(
                network_swinir=module,
                __path__=[str(candidates[0].parent.parent)],
            )
            spec.loader.exec_module(module)
            net = module.SwinIR

        # Model configuration based on type (align with SwinIR main_test_swinir.define_model where applicable)
        model_path_resolved = self._get_default_model_path()

        if self.model_type == 'classical':
            ck_upscale, ck_tps = _parse_swinir_classical_checkpoint(
                model_path_resolved if model_path_resolved else Path("_dummy_x4_s48w8_.pth")
            )
            if self.scale != ck_upscale and model_path_resolved and model_path_resolved.exists():
                logger.warning(
                    "SwinIR wrapper scale=%s does not match checkpoint name (x%s); using checkpoint upscale for the graph.",
                    self.scale,
                    ck_upscale,
                )
            effective_scale = ck_upscale if model_path_resolved and model_path_resolved.exists() else self.scale
            self._effective_upscale = effective_scale
            tps = self.training_patch_size if self.training_patch_size is not None else ck_tps
            self.model = net(
                upscale=effective_scale,
                in_chans=3,
                img_size=tps,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='pixelshuffle',
                resi_connection='1conv',
            )

        elif self.model_type == 'lightweight':
            self._effective_upscale = self.scale
            self.model = net(
                upscale=self.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='pixelshuffledirect',
                resi_connection='1conv',
            )

        elif self.model_type == 'real_sr':
            self._effective_upscale = self.scale
            self.model = net(
                upscale=self.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='nearest+conv',
                resi_connection='1conv',
            )

        elif self.model_type == 'denoise_grayscale':
            self._effective_upscale = 1
            self.model = net(
                upscale=1,
                in_chans=1,
                img_size=128,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='',
                resi_connection='1conv',
            )

        elif self.model_type == 'denoise_color':
            self._effective_upscale = 1
            self.model = net(
                upscale=1,
                in_chans=3,
                img_size=128,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='',
                resi_connection='1conv',
            )

        elif self.model_type == 'jpeg':
            self._effective_upscale = 1
            self.model = net(
                upscale=1,
                in_chans=3,
                img_size=126,
                window_size=7,
                img_range=255.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2.0,
                upsampler='',
                resi_connection='1conv',
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load pretrained weights
        model_path = self._get_default_model_path()
        if model_path and model_path.exists():
            logger.info(f"Loading SwinIR weights from {model_path}")
            try:
                loaded_net = torch.load(model_path, map_location='cpu')
                if 'params' in loaded_net:
                    state_dict = loaded_net['params']
                elif 'params_ema' in loaded_net:
                    state_dict = loaded_net['params_ema']
                else:
                    state_dict = loaded_net
                
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info("SwinIR weights loaded successfully (strict)")
                except RuntimeError as e:
                    load_result = self.model.load_state_dict(state_dict, strict=False)
                    logger.warning(
                        "SwinIR strict load failed (%s); partial load: %s missing, %s unexpected",
                        e,
                        len(load_result.missing_keys),
                        len(load_result.unexpected_keys),
                    )
            except Exception as e:
                logger.warning(f"Failed to load SwinIR weights: {e}. Using random initialization.")
        else:
            logger.warning(f"No pretrained weights found at {model_path}. Using random initialization.")

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"SwinIR model loaded on {self.device}")

    def _tile_process(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Process image in tiles to handle large images.

        Args:
            input_tensor: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H', W']
        """
        if self.tile_size <= 0 or input_tensor.shape[2] * input_tensor.shape[3] < 256 * 256:
            # Process full image
            return self._forward_pass(input_tensor)

        # Tile-based processing
        B, C, H, W = input_tensor.shape
        tile_size = self.tile_size
        pad = self.tile_pad

        # Calculate padding
        mod_pad_h = (tile_size - H % tile_size) % tile_size
        mod_pad_w = (tile_size - W % tile_size) % tile_size

        # Pad input
        input_padded = torch.cat([input_tensor, torch.flip(input_tensor, [2, 3])], dim=0)
        input_padded = torch.cat([input_padded, input_padded[:, :, :, :1]], dim=3)
        input_padded = torch.cat([input_padded, input_padded[:, :, :1, :]], dim=2)

        # Process each tile
        output_padded = torch.zeros_like(input_padded)

        stride = tile_size - 2 * pad
        h_steps = max(1, (H - tile_size) // stride + 1)
        w_steps = max(1, (W - tile_size) // stride + 1)

        for h_idx in range(h_steps + 1):
            for w_idx in range(w_steps + 1):
                h_start = h_idx * stride
                w_start = w_idx * stride
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)

                # Get tile with padding
                tile = input_padded[:, :, h_start:h_end + pad, w_start:w_end + pad]

                # Process tile
                tile_out = self._forward_pass(tile)

                # Remove padding and accumulate
                out_h_start = h_start
                out_w_start = w_start
                out_h_end = out_h_start + (h_end - h_start)
                out_w_end = out_w_start + (w_end - w_start)

                output_padded[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_out

        # Crop to original size
        output = output_padded[:, :, :H, :W]

        eff = getattr(self, '_effective_upscale', self.scale)
        if eff > 1 and self.model_type in ['classical', 'lightweight', 'real_sr']:
            output = torch.nn.functional.interpolate(
                output,
                size=(H * eff, W * eff),
                mode='bicubic',
                align_corners=False
            )

        return output

    def _forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Single forward pass without tiling."""
        with torch.no_grad():
            if self.model is None:
                self._load_model()

            input_tensor = input_tensor.to(self.device)

            if self.model_type in ['classical', 'lightweight', 'real_sr']:
                # SR models: output has different size
                output = self.model(input_tensor)
            else:
                # Denoise/JPEG models: same size output
                output = self.model(input_tensor)

            return output

    def restore(
        self,
        degraded_tensor: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Restore degraded image using SwinIR.

        Args:
            degraded_tensor: Input tensor [B, C, H, W], range [0, 1]
            return_dict: If True, return dict with metadata

        Returns:
            Restored tensor [B, C, H', W'] or dict with tensor and metadata

        Raises:
            ValueError: If input tensor format is invalid
        """
        # Validate input
        if degraded_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {degraded_tensor.dim()}D")

        B, C, H, W = degraded_tensor.shape

        # Ensure 3 channels (RGB)
        if C == 1:
            # Convert grayscale to RGB
            degraded_tensor = degraded_tensor.repeat(1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {C}")

        logger.debug(f"SwinIR processing: {B}x{C}x{H}x{W}")

        # Run inference
        try:
            # SwinIR classical models expect BGR (OpenCV convention)
            # Convert RGB->BGR for model input, then BGR->RGB for output
            degraded_tensor_bgr = degraded_tensor.flip(1) if degraded_tensor.shape[1] == 3 else degraded_tensor

            restored = self._tile_process(degraded_tensor_bgr)
            restored = torch.clamp(restored, 0.0, 1.0)

            # Convert BGR output back to RGB
            if restored.shape[1] == 3:
                restored = restored.flip(1)

            if return_dict:
                return {
                    'restored': restored,
                    'model_name': 'SwinIR',
                    'model_type': self.model_type,
                    'scale': self.scale,
                    'device': self.device,
                    'input_shape': degraded_tensor.shape,
                    'output_shape': restored.shape,
                }

            return restored

        except Exception as e:
            logger.error(f"SwinIR inference failed: {e}")
            raise

    def unload(self) -> None:
        """
        Unload model and free GPU memory.

        Call this after inference to release VRAM.
        """
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info("SwinIR model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.unload()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False


def load_swinir(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    model_type: str = 'classical',
    scale: int = 4,
    **kwargs
) -> SwinIRWrapper:
    """
    Factory function to create and load SwinIR wrapper.

    Args:
        model_path: Path to pretrained weights
        device: Computation device
        model_type: Type of SwinIR model
        scale: Upscaling factor
        **kwargs: Additional arguments

    Returns:
        Loaded SwinIRWrapper instance
    """
    wrapper = SwinIRWrapper(
        model_path=model_path,
        device=device,
        model_type=model_type,
        scale=scale,
        **kwargs
    )
    wrapper._load_model()
    return wrapper


# ============================================================================
# Example usage
# ============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create wrapper
    swinir = SwinIRWrapper(model_type='classical', scale=4)

    # Test with random input
    test_input = torch.rand(1, 3, 128, 128)

    # Restore
    result = swinir.restore(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {result.shape}")

    # Cleanup
    swinir.unload()
