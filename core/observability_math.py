"""
Observability Map Generation Module
====================================
Core mathematical logic for computing visual observability maps.

The observability map quantifies how well different image regions can be
recovered based on consensus across multiple restoration models.

Formula: O = exp(-(alpha * Error + beta * Variance))

Author: Visual Observability Benchmark Team
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


def generate_observability_map(
    gt_tensor: torch.Tensor,
    restored_list: List[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 0.5,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, dict]:
    """
    Generate observability map based on restoration consensus.

    This function computes an observability map that indicates how well
    different image regions can be recovered, based on:
    1. Error: How far the consensus mean is from the ground truth
    2. Variance: How much disagreement exists across restoration models

    Higher observability = easier to recover (low error, high model agreement)
    Lower observability = harder to recover (high error, low model agreement)

    Args:
        gt_tensor: Ground truth tensor [B, C, H, W], range [0, 1]
        restored_list: List of K restored tensors, each [B, C, H, W], range [0, 1]
        alpha: Weight for error term (default: 1.0)
        beta: Weight for variance term (default: 0.5)
        eps: Small constant to prevent numerical instability (default: 1e-8)

    Returns:
        Tuple containing:
            - observability_map: Tensor [B, 1, H, W], range [0, 1]
              Higher values indicate better observability (easier to recover)
            - stats_dict: Dictionary containing intermediate statistics:
                - 'error_map': MSE between consensus mean and GT
                - 'variance_map': Variance across restoration models
                - 'consensus_mean': Average of restored images
                - 'num_models': Number of models in consensus

    Raises:
        ValueError: If input tensors have mismatched dimensions
        ValueError: If restored_list is empty
    """
    # ==========================================================================
    # Input Validation and Normalization
    # ==========================================================================
    if gt_tensor.dim() != 4:
        raise ValueError(f"gt_tensor must be 4D [B, C, H, W], got {gt_tensor.dim()}D")

    if len(restored_list) == 0:
        raise ValueError("restored_list cannot be empty")

    B, C, H, W = gt_tensor.shape

    # Ensure gt_tensor is in [0, 1] range
    gt_tensor = torch.clamp(gt_tensor, 0.0, 1.0)

    # Resize and validate all restored tensors to match GT shape
    processed_restored = []
    for i, rest_tensor in enumerate(restored_list):
        # Ensure tensor is 4D
        if rest_tensor.dim() == 3:
            rest_tensor = rest_tensor.unsqueeze(0)
        if rest_tensor.dim() == 5:
            rest_tensor = rest_tensor.squeeze(0)

        # Resize to match GT
        if rest_tensor.shape != gt_tensor.shape:
            rest_tensor = F.interpolate(rest_tensor, size=(H, W), mode='bicubic', align_corners=False)

        # Ensure in [0, 1] range
        rest_tensor = torch.clamp(rest_tensor, 0.0, 1.0)
        processed_restored.append(rest_tensor)

    restored_list = processed_restored

    K = len(restored_list)

    # ==========================================================================
    # Step 1: Compute Consensus Mean
    # ==========================================================================
    # Stack all restored tensors and compute mean along model dimension
    restored_stack = torch.stack(restored_list, dim=0)  # [K, B, C, H, W]
    consensus_mean = torch.mean(restored_stack, dim=0)  # [B, C, H, W]

    # ==========================================================================
    # Step 2: Compute Error Map (MSE between Consensus Mean and GT)
    # ==========================================================================
    # Calculate per-pixel squared error
    squared_error = (consensus_mean - gt_tensor) ** 2  # [B, C, H, W]

    # Mean across channels to get single-channel error map
    error_map = torch.mean(squared_error, dim=1, keepdim=True)  # [B, 1, H, W]

    # Ensure no NaN in error map
    error_map = torch.nan_to_num(error_map, nan=1.0)

    # ==========================================================================
    # Step 3: Compute Variance Map (across restoration models)
    # ==========================================================================
    # Calculate variance across model dimension (dimension 0 of stacked tensor)
    # Var = E[X^2] - E[X]^2
    restored_squared_mean = torch.mean(restored_stack ** 2, dim=0)  # [B, C, H, W]
    variance_per_channel = restored_squared_mean - (consensus_mean ** 2)  # [B, C, H, W]

    # Mean across channels for unified variance map
    variance_map = torch.mean(variance_per_channel, dim=1, keepdim=True)  # [B, 1, H, W]

    # Ensure variance is non-negative (numerical safety)
    variance_map = torch.clamp(variance_map, min=0.0)

    # Ensure no NaN in variance map
    variance_map = torch.nan_to_num(variance_map, nan=0.0)

    # ==========================================================================
    # Step 4: Compute Observability Map
    # ==========================================================================
    # Formula: O = exp(-(alpha * Error + beta * Variance))
    combined_metric = alpha * error_map + beta * variance_map
    observability_map = torch.exp(-combined_metric)

    # Ensure observability is in valid range [0, 1]
    observability_map = torch.clamp(observability_map, 0.0, 1.0)

    # ==========================================================================
    # Step 5: Compile Statistics
    # ==========================================================================
    stats_dict = {
        'error_map': error_map,                          # [B, 1, H, W]
        'variance_map': variance_map,                    # [B, 1, H, W]
        'consensus_mean': consensus_mean,               # [B, C, H, W]
        'num_models': K,
        'alpha': alpha,
        'beta': beta,
        'mean_error': torch.nanmean(error_map).item(),
        'mean_variance': torch.nanmean(variance_map).item(),
        'mean_observability': torch.nanmean(observability_map).item(),
    }

    return observability_map, stats_dict


def generate_observability_map_adaptive(
    gt_tensor: torch.Tensor,
    restored_list: List[torch.Tensor],
    base_alpha: float = 1.0,
    base_beta: float = 0.5,
    percentile_based: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Generate adaptive observability map with automatic parameter tuning.

    Uses percentile-based scaling for alpha and beta based on the
    distribution of errors and variances across the image.

    Args:
        gt_tensor: Ground truth tensor [B, C, H, W]
        restored_list: List of restored tensors
        base_alpha: Base weight for error term
        base_beta: Base weight for variance term
        percentile_based: If True, scale weights based on error/variance distribution

    Returns:
        Tuple of (observability_map, stats_dict) with additional adaptive metrics
    """
    # First compute basic maps
    obs_map, stats = generate_observability_map(
        gt_tensor, restored_list,
        alpha=base_alpha, beta=base_beta
    )

    if percentile_based:
        # Compute percentile-based scaling
        error_map = stats['error_map']
        variance_map = stats['variance_map']

        # Find 90th percentile values for adaptive scaling
        error_90th = torch.quantile(error_map, 0.9).item()
        variance_90th = torch.quantile(variance_map, 0.9).item()

        # Normalize maps using percentiles
        error_normalized = error_map / (error_90th + 1e-8)
        variance_normalized = variance_map / (variance_90th + 1e-8)

        # Compute adaptive observability
        combined = base_alpha * error_normalized + base_beta * variance_normalized
        obs_map_adaptive = torch.exp(-combined)
        obs_map_adaptive = torch.clamp(obs_map_adaptive, 0.0, 1.0)

        # Add adaptive statistics
        stats['error_normalized'] = error_normalized
        stats['variance_normalized'] = variance_normalized
        stats['error_90th_percentile'] = error_90th
        stats['variance_90th_percentile'] = variance_90th
        stats['observability_adaptive'] = obs_map_adaptive

    return obs_map, stats


def compute_local_observability(
    gt_tensor: torch.Tensor,
    restored_list: List[torch.Tensor],
    window_size: int = 8,
    stride: int = 4,
    alpha: float = 1.0,
    beta: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """
    Compute local observability using patch-based analysis.

    Divides the image into local patches and computes observability
    for each patch, then interpolates back to full resolution.

    Args:
        gt_tensor: Ground truth tensor [B, C, H, W]
        restored_list: List of restored tensors
        window_size: Size of local window/patch
        stride: Stride for patch sampling
        alpha: Weight for error term
        beta: Weight for variance term

    Returns:
        Tuple of (local_observability_map, local_stats_dict)
    """
    B, C, H, W = gt_tensor.shape

    # Extract patches using unfold
    patches_gt = F.unfold(gt_tensor, kernel_size=window_size, stride=stride)  # [B, C*k^2, N]
    N = patches_gt.shape[-1]
    patches_gt = patches_gt.view(B, C, window_size, window_size, N)  # [B, C, k, k, N]
    patches_gt = patches_gt.permute(0, 4, 1, 2, 3)  # [B, N, C, k, k]

    # Process each patch through the full pipeline
    local_observability = []
    local_errors = []
    local_variances = []

    for n in range(N):
        patch_gt = patches_gt[:, n:n+1, :, :, :]  # [B, 1, C, k, k]
        patch_restored = [
            F.unfold(r, kernel_size=window_size, stride=stride)[:, :, n:n+1].view(B, C, window_size, window_size)
            for r in restored_list
        ]
        patch_restored = [p.unsqueeze(1) for p in patch_restored]  # [B, 1, C, k, k]

        # Reshape for observability computation
        patch_gt_flat = patch_gt.flatten(2).permute(0, 2, 1).view(B, 1, C, window_size, window_size)
        patch_restored_flat = [
            p.flatten(2).permute(0, 2, 1).view(B, 1, C, window_size, window_size)
            for p in patch_restored
        ]

        # This is simplified; for full implementation, adapt observability logic
        obs, stats = generate_observability_map(
            patch_gt_flat.squeeze(1), [p.squeeze(1) for p in patch_restored_flat],
            alpha=alpha, beta=beta
        )

        local_observability.append(obs.mean().item())
        local_errors.append(stats['mean_error'])
        local_variances.append(stats['mean_variance'])

    # Reshape back to spatial coordinates
    out_h = (H - window_size) // stride + 1
    out_w = (W - window_size) // stride + 1

    local_obs_tensor = torch.tensor(local_observability, device=gt_tensor.device).view(1, 1, out_h, out_w)

    # Upsample to original resolution
    local_observability_full = F.interpolate(
        local_obs_tensor, size=(H, W), mode='bicubic', align_corners=False
    )

    stats['patch_size'] = window_size
    stats['num_patches'] = N
    stats['local_observability_raw'] = local_obs_tensor

    return local_observability_full, stats


def compute_observability_gradient(
    observability_map: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradient/edge map from observability map.

    Higher gradient values indicate rapid changes in observability,
    which may correspond to important structural boundaries.

    Args:
        observability_map: Observability tensor [B, 1, H, W]

    Returns:
        Gradient magnitude map [B, 1, H, W]
    """
    # Sobel kernels for gradient computation
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=observability_map.dtype, device=observability_map.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=observability_map.dtype, device=observability_map.device).view(1, 1, 3, 3)

    # Apply Sobel filters
    grad_x = F.conv2d(observability_map, sobel_x, padding=1)
    grad_y = F.conv2d(observability_map, sobel_y, padding=1)

    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return gradient_magnitude


def compute_region_statistics(
    observability_map: torch.Tensor,
    gt_tensor: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Compute region-based statistics from observability map.

    Classifies regions into high/low observability based on threshold.

    Args:
        observability_map: Observability tensor [B, 1, H, W]
        gt_tensor: Ground truth tensor [B, C, H, W]
        threshold: Threshold for classifying high/low observability

    Returns:
        Dictionary with region statistics
    """
    B = observability_map.shape[0]

    stats = {
        'global_mean': torch.mean(observability_map).item(),
        'global_std': torch.std(observability_map).item(),
        'global_min': torch.min(observability_map).item(),
        'global_max': torch.max(observability_map).item(),
    }

    # Per-batch statistics
    batch_stats = []
    for b in range(B):
        obs_b = observability_map[b]
        gt_b = gt_tensor[b]

        # High vs low observability regions
        high_obs_mask = obs_b > threshold
        low_obs_mask = ~high_obs_mask

        high_obs_ratio = high_obs_mask.float().mean().item()
        low_obs_ratio = low_obs_mask.float().mean().item()

        # Mean GT intensity in each region
        mean_gt_high = gt_b[:, high_obs_mask.squeeze(0)].mean().item() if high_obs_mask.any() else 0.0
        mean_gt_low = gt_b[:, low_obs_mask.squeeze(0)].mean().item() if low_obs_mask.any() else 0.0

        batch_stats.append({
            'batch_id': b,
            'high_observability_ratio': high_obs_ratio,
            'low_observability_ratio': low_obs_ratio,
            'mean_gt_in_high_obs_region': mean_gt_high,
            'mean_gt_in_low_obs_region': mean_gt_low,
        })

    stats['per_batch'] = batch_stats

    return stats


class ObservabilityMetrics:
    """
    Utility class for computing and tracking observability metrics.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.history = []

    def compute(self, gt_tensor: torch.Tensor, restored_list: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """Compute observability map and metrics."""
        obs_map, stats = generate_observability_map(
            gt_tensor, restored_list, alpha=self.alpha, beta=self.beta
        )

        # Add region statistics
        region_stats = compute_region_statistics(obs_map, gt_tensor)
        stats.update(region_stats)

        # Store in history
        self.history.append(stats)

        return obs_map, stats

    def compute_batch_summary(self) -> dict:
        """Compute summary statistics across all computed samples."""
        if not self.history:
            return {}

        mean_errors = [h['mean_error'] for h in self.history]
        mean_variances = [h['mean_variance'] for h in self.history]
        mean_observabilities = [h['mean_observability'] for h in self.history]

        return {
            'total_samples': len(self.history),
            'avg_error': np.mean(mean_errors),
            'std_error': np.std(mean_errors),
            'avg_variance': np.mean(mean_variances),
            'std_variance': np.std(mean_variances),
            'avg_observability': np.mean(mean_observabilities),
            'std_observability': np.std(mean_observabilities),
        }

    def reset_history(self):
        """Clear computation history."""
        self.history = []
