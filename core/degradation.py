"""
Image Degradation Engine
========================
Provides various image degradation methods for synthetic dataset generation.
Supports degradation models: classical SR, real-world degradation, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class DegradationEngine:
    """
    Image degradation engine for generating synthetic low-quality images.

    Supports multiple degradation models:
    - Classical degradation: blur -> noise -> downsample -> upsample -> noise
    - Real-world degradation: more complex pipeline with randomized operations

    Args:
        scale: Upscaling factor (default: 4)
        degradation_type: Type of degradation ('classical' or 'realworld')
        use_gaussian_blur: Whether to apply Gaussian blur
        use_jpeg: Whether to simulate JPEG compression artifact
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        scale: int = 4,
        degradation_type: str = 'classical',
        use_gaussian_blur: bool = True,
        use_jpeg: bool = False,
        seed: Optional[int] = None
    ):
        self.scale = scale
        self.degradation_type = degradation_type
        self.use_gaussian_blur = use_gaussian_blur
        self.use_jpeg = use_jpeg

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        self.kernel_options = self._init_kernels()

    def _init_kernels(self) -> dict:
        """Initialize available blur kernels."""
        return {
            'iso': self._generate_iso_kernel,
            'aniso': self._generate_aniso_kernel,
            'disk': self._generate_disk_kernel,
        }

    def _generate_iso_kernel(self, sigma: float = 2.0) -> torch.Tensor:
        """Generate isotropic Gaussian kernel."""
        size = int(2 * 3 * sigma + 1)
        size = size if size % 2 == 1 else size + 1
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        return kernel_2d

    def _generate_aniso_kernel(self, sigma_x: float = 2.0, sigma_y: float = 6.0) -> torch.Tensor:
        """Generate anisotropic Gaussian kernel."""
        size_x = int(2 * 3 * sigma_x + 1)
        size_y = int(2 * 3 * sigma_y + 1)
        size_x = size_x if size_x % 2 == 1 else size_x + 1
        size_y = size_y if size_y % 2 == 1 else size_y + 1

        x = torch.arange(-(size_x // 2), size_x // 2 + 1, dtype=torch.float32)
        y = torch.arange(-(size_y // 2), size_y // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        gauss_x = torch.exp(-(xx ** 2) / (2 * sigma_x ** 2))
        gauss_y = torch.exp(-(yy ** 2) / (2 * sigma_y ** 2))
        kernel_2d = gauss_x * gauss_y
        return kernel_2d / kernel_2d.sum()

    def _generate_disk_kernel(self, radius: float = 3.0) -> torch.Tensor:
        """Generate disk (pillbox) kernel."""
        size = int(2 * radius + 1)
        size = size if size % 2 == 1 else size + 1
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel_2d = ((xx ** 2 + yy ** 2) <= radius ** 2).float()
        return kernel_2d / kernel_2d.sum()

    def _generate_random_kernel(self) -> Tuple[torch.Tensor, List[float]]:
        """Generate a random blur kernel with random parameters."""
        kernel_types = list(self.kernel_options.keys())
        kernel_type = self.rng.choice(kernel_types)

        if kernel_type == 'iso':
            sigma = self.rng.uniform(0.1, 4.0)
            kernel = self.kernel_options['iso'](sigma)
            params = [sigma]
        elif kernel_type == 'aniso':
            sigma_x = self.rng.uniform(0.1, 2.5)
            sigma_y = self.rng.uniform(2.5, 6.0)
            kernel = self.kernel_options['aniso'](sigma_x, sigma_y)
            params = [sigma_x, sigma_y]
        else:  # disk
            radius = self.rng.uniform(1.5, 4.5)
            kernel = self.kernel_options['disk'](radius)
            params = [radius]

        return kernel, params

    def _add_gaussian_noise(self, tensor: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def _add_poisson_noise(self, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Add Poisson noise to tensor (shot noise simulation)."""
        vals = len(self.rng.poisson(1, (1,)))
        arr = tensor.cpu().numpy() * 255.0 * scale
        noisy = self.rng.poisson(arr) / (255.0 * scale)
        return torch.from_numpy(np.clip(noisy, 0.0, 1.0)).to(tensor.device).type(tensor.dtype)

    def _apply_blur(self, tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply 2D convolution blur to tensor."""
        if tensor.dim() == 4:
            B, C, H, W = tensor.shape
            kernel = kernel.to(tensor.device)

            # Handle RGB and grayscale
            if C == 3:
                kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3, -1, -1, -1)
            elif C == 1:
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            else:
                kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)

            padding = kernel.shape[-1] // 2
            blurred = F.conv2d(tensor, kernel, padding=padding, groups=C)
            return blurred
        else:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {tensor.dim()}D")

    def _downsample(self, tensor: torch.Tensor, scale: int) -> torch.Tensor:
        """Downsample tensor by scale factor."""
        B, C, H, W = tensor.shape
        new_H, new_W = H // scale, W // scale
        return F.interpolate(tensor, size=(new_H, new_W), mode='bicubic', align_corners=False)

    def _upsample(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Upsample tensor to target size."""
        return F.interpolate(tensor, size=size, mode='bicubic', align_corners=False)

    def _simulate_jpeg_artifacts(self, tensor: torch.Tensor, quality: int = 80) -> torch.Tensor:
        """
        Simulate JPEG compression artifacts.

        Note: This is a simplified approximation. For accurate JPEG simulation,
        use torchvision's JPEG implementation or actual image encoding.
        """
        # Convert to numpy for JPEG simulation
        if tensor.is_cuda:
            tensor = tensor.cpu()

        arr = tensor.numpy()
        B, C, H, W = arr.shape

        for b in range(B):
            for c in range(C):
                img = arr[b, c]
                # Simple DCT-based artifact simulation (approximation)
                # Apply slight blurring based on quality
                blur_sigma = (100 - quality) / 100.0 * 1.5
                if blur_sigma > 0.1:
                    from scipy.ndimage import gaussian_filter
                    arr[b, c] = gaussian_filter(img, sigma=blur_sigma)

        # Add quantization-like noise
        arr = arr + np.random.randn(*arr.shape) * (100 - quality) / 10000.0
        arr = np.clip(arr, 0.0, 1.0)

        return torch.from_numpy(arr)

    def degrade(
        self,
        gt_tensor: torch.Tensor,
        blur_sigma: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        kernel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply degradation to ground truth image tensor.

        Args:
            gt_tensor: Input tensor [B, C, H, W] in range [0, 1]
            blur_sigma: Custom blur sigma (if None, random)
            noise_sigma: Custom noise level (if None, random)
            kernel: Custom blur kernel (if None, generated randomly)

        Returns:
            Tuple of (degraded_tensor, metadata_dict)
        """
        if gt_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {gt_tensor.shape}")

        device = gt_tensor.device
        degraded = gt_tensor.clone()

        # Store metadata
        metadata = {
            'scale': self.scale,
            'degradation_type': self.degradation_type,
        }

        # Step 1: Apply blur (if enabled)
        if self.use_gaussian_blur:
            if kernel is None:
                kernel, kernel_params = self._generate_random_kernel()
            else:
                kernel_params = []

            kernel = kernel.to(device)
            metadata['kernel_type'] = type(kernel).__name__
            metadata['kernel_params'] = kernel_params

            degraded = self._apply_blur(degraded, kernel)

        # Step 2: Add noise
        if noise_sigma is None:
            noise_sigma = self.rng.uniform(0.0, 0.05)  # 0 ~ 5% noise

        metadata['noise_sigma'] = noise_sigma

        if noise_sigma > 0:
            # Randomly choose noise type
            noise_type = self.rng.choice(['gaussian', 'poisson', 'both'], p=[0.7, 0.15, 0.15])
            metadata['noise_type'] = noise_type

            if 'gaussian' in noise_type:
                degraded = self._add_gaussian_noise(degraded, noise_sigma)
            if 'poisson' in noise_type:
                degraded = self._add_poisson_noise(degraded)

        # Step 3: Downsample and upsample (for classic SR)
        if self.scale > 1:
            B, C, H, W = degraded.shape
            downsampled = self._downsample(degraded, self.scale)
            upsampled = self._upsample(downsampled, (H, W))
            degraded = upsampled
            metadata['downsampled_size'] = (H // self.scale, W // self.scale)

        # Step 4: JPEG artifact simulation (if enabled)
        if self.use_jpeg:
            jpeg_quality = self.rng.randint(60, 95)
            metadata['jpeg_quality'] = jpeg_quality
            # Note: Simplified simulation, consider using actual JPEG encoding for accuracy

        # Ensure output is in valid range
        degraded = torch.clamp(degraded, 0.0, 1.0)

        return degraded, metadata

    def degrade_batch(self, gt_tensors: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, List[dict]]:
        """
        Apply degradation to a batch of images.

        Args:
            gt_tensors: Batch of tensors [B, C, H, W]
            **kwargs: Additional arguments passed to degrade()

        Returns:
            Tuple of (degraded_batch, metadata_list)
        """
        degraded_list = []
        metadata_list = []

        for i in range(gt_tensors.shape[0]):
            degraded, metadata = self.degrade(gt_tensors[i:i+1], **kwargs)
            degraded_list.append(degraded)
            metadata_list.append(metadata)

        degraded_batch = torch.cat(degraded_list, dim=0)
        return degraded_batch, metadata_list

    def get_degradation_preset(self, preset_name: str) -> dict:
        """
        Get predefined degradation parameters.

        Args:
            preset_name: Name of preset ('light', 'medium', 'heavy', 'extreme')

        Returns:
            Dictionary of degradation parameters
        """
        presets = {
            'light': {
                'blur_sigma': 1.0,
                'noise_sigma': 0.005,
                'jpeg_quality': 90,
            },
            'medium': {
                'blur_sigma': 2.0,
                'noise_sigma': 0.015,
                'jpeg_quality': 80,
            },
            'heavy': {
                'blur_sigma': 3.5,
                'noise_sigma': 0.03,
                'jpeg_quality': 70,
            },
            'extreme': {
                'blur_sigma': 5.0,
                'noise_sigma': 0.05,
                'jpeg_quality': 60,
            },
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        return presets[preset_name]


class RealWorldDegradationEngine(DegradationEngine):
    """
    Real-world degradation engine with more complex pipeline.

    Simulates authentic degradations found in real-world images:
    - Motion blur
    - Out-of-focus blur
    - Multiple noise types
    - Sensor artifacts
    - Multiple JPEG re-compression
    """

    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(scale=scale, degradation_type='realworld', **kwargs)

    def _generate_motion_blur_kernel(self, length: float = 10.0, angle: float = 45.0) -> torch.Tensor:
        """Generate motion blur kernel."""
        size = int(length * 2)
        size = size if size % 2 == 1 else size + 1

        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        angle_rad = torch.tensor(angle * np.pi / 180.0)
        x_rot = xx * torch.cos(angle_rad) + yy * torch.sin(angle_rad)

        kernel = torch.exp(-(x_rot ** 2) / (2 * (length / 3) ** 2))
        return kernel / kernel.sum()

    def degrade_realworld(self, gt_tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply real-world degradation pipeline.

        Args:
            gt_tensor: Input tensor [B, C, H, W] in range [0, 1]

        Returns:
            Tuple of (degraded_tensor, metadata)
        """
        if gt_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {gt_tensor.shape}")

        device = gt_tensor.device
        degraded = gt_tensor.clone()
        metadata = {'degradation_type': 'realworld', 'scale': self.scale}

        # Step 1: Random blur (motion or Gaussian)
        blur_type = self.rng.choice(['gaussian', 'motion', 'aniso'])
        metadata['blur_type'] = blur_type

        if blur_type == 'gaussian':
            sigma = self.rng.uniform(0.5, 3.0)
            kernel = self._generate_iso_kernel(sigma)
        elif blur_type == 'motion':
            length = self.rng.uniform(5, 15)
            angle = self.rng.uniform(0, 360)
            kernel = self._generate_motion_blur_kernel(length, angle)
        else:
            sigma_x = self.rng.uniform(0.5, 2.0)
            sigma_y = self.rng.uniform(3.0, 7.0)
            kernel = self._generate_aniso_kernel(sigma_x, sigma_y)

        degraded = self._apply_blur(degraded, kernel.to(device))

        # Step 2: Multiple noise types
        # Gaussian noise
        if self.rng.random() < 0.8:
            gauss_sigma = self.rng.uniform(0.005, 0.03)
            degraded = self._add_gaussian_noise(degraded, gauss_sigma)
            metadata['gaussian_noise'] = gauss_sigma

        # Poisson noise
        if self.rng.random() < 0.5:
            degraded = self._add_poisson_noise(degraded)
            metadata['poisson_noise'] = True

        # Step 3: Downsample/Upsample
        if self.scale > 1:
            B, C, H, W = degraded.shape
            downsampled = self._downsample(degraded, self.scale)

            # Add noise after downsample (simulating sensor noise at low-res)
            if self.rng.random() < 0.5:
                sensor_noise = self.rng.uniform(0.005, 0.02)
                downsampled = self._add_gaussian_noise(downsampled, sensor_noise)

            upsampled = self._upsample(downsampled, (H, W))
            degraded = upsampled

        # Step 4: JPEG compression simulation (multiple times)
        num_compressions = self.rng.randint(1, 3)
        metadata['num_compressions'] = num_compressions

        for i in range(num_compressions):
            quality = self.rng.randint(60, 95)
            degraded = self._simulate_jpeg_artifacts(degraded, quality)

        # Step 5: Final noise pass
        if self.rng.random() < 0.3:
            final_noise = self.rng.uniform(0.005, 0.015)
            degraded = self._add_gaussian_noise(degraded, final_noise)
            metadata['final_noise'] = final_noise

        # Ensure valid range
        degraded = torch.clamp(degraded, 0.0, 1.0)

        return degraded, metadata


def create_degradation_engine(
    engine_type: str = 'classical',
    scale: int = 4,
    **kwargs
) -> DegradationEngine:
    """
    Factory function to create degradation engine.

    Args:
        engine_type: Type of engine ('classical' or 'realworld')
        scale: Upscaling factor
        **kwargs: Additional arguments for engine

    Returns:
        DegradationEngine instance
    """
    if engine_type == 'classical':
        return DegradationEngine(scale=scale, **kwargs)
    elif engine_type == 'realworld':
        return RealWorldDegradationEngine(scale=scale, **kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Use 'classical' or 'realworld'")
