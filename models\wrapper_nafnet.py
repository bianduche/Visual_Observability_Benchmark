"""
NAFNet Model Wrapper
=====================
Wrapper interface for NAFNet (Simple Baseline for Image Restoration).

NAFNet is a simple yet effective network for image denoising and deblurring.
Uses the OFFICIAL NAFNet architecture (from NAFNet paper) to ensure weight
compatibility with pretrained checkpoints.

Reference: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, List
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Official GoPro NAFNet-width32 layout (options/test/GoPro/NAFNet-width32.yml)
GOPRO_NAFNET_WIDTH = 32
GOPRO_ENC_BLK_NUMS = [1, 1, 1, 28]
GOPRO_MIDDLE_BLK_NUM = 1
GOPRO_DEC_BLK_NUMS = [1, 1, 1, 1]


# ─────────────────────────────────────────────────────────────
#  OFFICIAL NAFNet implementation (inlined, no import conflicts)
#  Source: NAFNet-main/NAFNet-main/basicsr/models/archs/NAFNet_arch.py
# ─────────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (official NAFNet implementation)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)


class SimpleGate(nn.Module):
    """Simple gate from NAFNet."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet block with gated convolution (official implementation)."""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, bias=True, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    Official NAFNet architecture (U-Net style with encoders/decoders).

    Matches the structure of NAFNet-GoPro-width32.pth weights exactly.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1,
                 enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Local_Base:
    """Placeholder for Local_Base mixin (not needed for inference)."""
    def convert(self, *args, **kwargs):
        pass


class NAFNetLocal(Local_Base, NAFNet):
    """NAFNet with local base (not needed for inference)."""
    pass


def _build_official_nafnet():
    """Return the official NAFNet class for GoPro-width32."""
    return NAFNet


# ─────────────────────────────────────────────────────────────
#  NAFNetWrapper class
# ─────────────────────────────────────────────────────────────

class NAFNetWrapper:
    """
    Wrapper class for NAFNet model.

    Provides a clean interface for running NAFNet inference with
    automatic model loading and memory management.

    Args:
        model_path: Path to pretrained NAFNet weights (.pth file)
        device: Computation device ('cuda' or 'cpu')
        width: Model width (number of base channels)
        n_blocks: Number of NAFBlocks
        task: Task type ('denoise', 'deblur', 'sr')
        scale: Upscaling factor for super-resolution (default: 1)
        tile_size: Tile size for tile-based inference (0 for full image)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        width: int = 32,
        n_blocks: int = 8,
        task: str = 'denoise',
        scale: int = 1,
        tile_size: int = 0,
        ensemble_mode: bool = False
    ):
        self.model_path = model_path
        self.width = width
        self.n_blocks = n_blocks
        self.task = task
        self.scale = scale
        self.tile_size = tile_size
        self.ensemble_mode = ensemble_mode

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
        nafnet_files = [
            # In models/nafnet/ folder
            base_path / 'models' / 'nafnet' / 'NAFNet-GoPro-width32.pth',
            base_path / 'models' / 'nafnet' / 'NAFNet-width32-nblocks8.pth',
            # In project root (direct download location)
            base_path / 'NAFNet-GoPro-width32.pth',
        ]

        for p in nafnet_files:
            if p.is_file():
                logger.info(f"Found NAFNet weights: {p}")
                return p

        return None

    def _load_model(self) -> None:
        """Load NAFNet model architecture and weights."""
        if self._loaded:
            return

        # Create model based on task (denoise/deblur: official U-Net NAFNet + GoPro-width32 weights)
        if self.task in ('denoise', 'deblur'):
            OfficialNAFNet = _build_official_nafnet()
            if self.n_blocks != 8 or self.width != GOPRO_NAFNET_WIDTH:
                logger.info(
                    "NAFNet GoPro 预训练与 width=%s enc/dec 布局绑定；忽略 n_blocks=%s，使用官方 [1,1,1,28] / middle=1 / dec [1,1,1,1]。",
                    GOPRO_NAFNET_WIDTH,
                    self.n_blocks,
                )
            self.model = OfficialNAFNet(
                img_channel=3,
                width=GOPRO_NAFNET_WIDTH,
                enc_blk_nums=GOPRO_ENC_BLK_NUMS,
                middle_blk_num=GOPRO_MIDDLE_BLK_NUM,
                dec_blk_nums=GOPRO_DEC_BLK_NUMS,
            )
        elif self.task == 'sr':
            # Super-resolution model
            self.model = NAFNetSR(
                in_chans=3,
                out_chans=3,
                dim=self.width,
                n_blocks=self.n_blocks,
                scale=self.scale
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # Load pretrained weights
        model_path = self._get_default_model_path()
        if model_path and model_path.exists():
            logger.info(f"Loading NAFNet weights from {model_path}")
            try:
                loaded_net = torch.load(model_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'params' in loaded_net:
                    state_dict = loaded_net['params']
                elif 'params_ema' in loaded_net:
                    state_dict = loaded_net['params_ema']
                else:
                    state_dict = loaded_net
                
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info("NAFNet weights loaded successfully (strict)")
                except RuntimeError as e:
                    load_result = self.model.load_state_dict(state_dict, strict=False)
                    logger.warning(
                        "NAFNet strict load failed (%s); partial load: %s missing, %s unexpected",
                        e,
                        len(load_result.missing_keys),
                        len(load_result.unexpected_keys),
                    )
            except Exception as e:
                logger.warning(f"Failed to load NAFNet weights: {e}. Using random initialization.")
        else:
            logger.warning(f"No pretrained weights found at {model_path}. Using random initialization.")

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"NAFNet model loaded on {self.device}")

    def _forward_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with no_grad."""
        with torch.no_grad():
            if self.model is None:
                self._load_model()

            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)

            return output

    def _tile_process(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process image in tiles for large images.

        Args:
            input_tensor: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H', W']
        """
        if self.tile_size <= 0 or input_tensor.shape[2] * input_tensor.shape[3] < 512 * 512:
            return self._forward_pass(input_tensor)

        # Tile-based processing
        B, C, H, W = input_tensor.shape
        tile_size = self.tile_size
        pad = 16

        # Calculate output size
        if self.task == 'sr':
            out_h, out_w = H * self.scale, W * self.scale
        else:
            out_h, out_w = H, W

        output = torch.zeros(B, C, out_h, out_w, device=input_tensor.device)

        stride = tile_size // 2
        h_steps = max(1, (H - tile_size) // stride + 1)
        w_steps = max(1, (W - tile_size) // stride + 1)

        for h_idx in range(h_steps + 1):
            for w_idx in range(w_steps + 1):
                h_start = min(h_idx * stride, H - tile_size)
                w_start = min(w_idx * stride, W - tile_size)

                # Get tile
                tile = input_tensor[:, :, h_start:h_start + tile_size, w_start:w_start + tile_size]

                # Process tile
                tile_out = self._forward_pass(tile)

                # Calculate output region
                if self.task == 'sr':
                    out_h_start = h_start * self.scale
                    out_w_start = w_start * self.scale
                    out_h_end = min(out_h_start + tile_size * self.scale, out_h)
                    out_w_end = min(out_w_start + tile_size * self.scale, out_w)
                    tile_h = out_h_end - out_h_start
                    tile_w = out_w_end - out_w_start
                    tile_out_crop = tile_out[:, :, :tile_h, :tile_w]
                else:
                    out_h_start = h_start
                    out_w_start = w_start
                    out_h_end = min(out_h_start + tile_size, out_h)
                    out_w_end = min(out_w_start + tile_size, out_w)
                    tile_h = out_h_end - out_h_start
                    tile_w = out_w_end - out_w_start
                    tile_out_crop = tile_out[:, :, :tile_h, :tile_w]

                # Accumulate (average overlapping regions)
                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end] = tile_out_crop

        return output

    def restore(
        self,
        degraded_tensor: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Restore degraded image using NAFNet.

        Args:
            degraded_tensor: Input tensor [B, C, H, W], range [0, 1]
            return_dict: If True, return dict with metadata

        Returns:
            Restored tensor [B, C, H', W'] or dict with tensor and metadata
        """
        # Validate input
        if degraded_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {degraded_tensor.dim()}D")

        B, C, H, W = degraded_tensor.shape

        # Ensure 3 channels (RGB)
        if C == 1:
            degraded_tensor = degraded_tensor.repeat(1, 3, 1, 1)
        elif C != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {C}")

        logger.debug(f"NAFNet processing: {B}x{C}x{H}x{W}")

        try:
            # Ensemble mode: process with multiple augmentations
            if self.ensemble_mode:
                restored = self._ensemble_inference(degraded_tensor)
            else:
                restored = self._tile_process(degraded_tensor)

            # Ensure output is in valid range
            restored = torch.clamp(restored, 0.0, 1.0)

            if return_dict:
                return {
                    'restored': restored,
                    'model_name': 'NAFNet',
                    'task': self.task,
                    'width': self.width,
                    'n_blocks': self.n_blocks,
                    'scale': self.scale,
                    'device': self.device,
                    'input_shape': degraded_tensor.shape,
                    'output_shape': restored.shape,
                }

            return restored

        except Exception as e:
            logger.error(f"NAFNet inference failed: {e}")
            raise

    def _ensemble_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensemble inference with test-time augmentation.

        Processes image with horizontal flip and averages results.
        """
        with torch.no_grad():
            # Original
            out1 = self._tile_process(input_tensor)

            # Horizontal flip
            input_flip = torch.flip(input_tensor, dims=[3])
            out2 = self._tile_process(input_flip)
            out2 = torch.flip(out2, dims=[3])

            # Average
            return (out1 + out2) / 2

    def unload(self) -> None:
        """
        Unload model and free GPU memory.
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

            logger.info("NAFNet model unloaded")

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


class NAFNetSR(nn.Module):
    """
    NAFNet for Super-Resolution.

    Uses pixel shuffle for upsampling instead of separate upsampling layer.
    """

    def __init__(
        self,
        in_chans: int = 3,
        out_chans: int = 3,
        dim: int = 32,
        n_blocks: int = 8,
        scale: int = 2
    ):
        super().__init__()

        self.scale = scale
        self.dim = dim

        # Initial projection - downsample first
        if scale > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chans, dim, kernel_size=3, padding=1),
                nn.PixelUnshuffle(scale),
            )
            in_chans_actual = dim * (scale ** 2)
        else:
            self.downsample = nn.Conv2d(in_chans, dim, kernel_size=3, padding=1)
            in_chans_actual = dim

        # Body
        self.body = nn.Sequential(*[
            NAFBlock(in_chans_actual, drop_out_rate=0.0)
            for _ in range(n_blocks)
        ])

        # Upsampling
        if scale > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_chans_actual, dim * (scale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)
            )
        else:
            self.upsample = nn.Conv2d(in_chans_actual, out_chans, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.body(x)
        x = self.upsample(x)
        return x


def load_nafnet(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    width: int = 32,
    n_blocks: int = 8,
    task: str = 'denoise',
    scale: int = 1,
    **kwargs
) -> NAFNetWrapper:
    """
    Factory function to create and load NAFNet wrapper.

    Args:
        model_path: Path to pretrained weights
        device: Computation device
        width: Model width
        n_blocks: Number of NAFBlocks
        task: Task type
        scale: Upscaling factor
        **kwargs: Additional arguments

    Returns:
        Loaded NAFNetWrapper instance
    """
    wrapper = NAFNetWrapper(
        model_path=model_path,
        device=device,
        width=width,
        n_blocks=n_blocks,
        task=task,
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
    nafnet = NAFNetWrapper(task='denoise', width=32, n_blocks=8)

    # Test with random input
    test_input = torch.rand(1, 3, 128, 128)

    # Restore
    result = nafnet.restore(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {result.shape}")

    # Cleanup
    nafnet.unload()
