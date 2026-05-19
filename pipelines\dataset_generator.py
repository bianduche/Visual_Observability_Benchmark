"""
Dataset Generator Pipeline
=========================
Main control logic for generating Visual Observability Benchmark datasets.

This module orchestrates:
1. Image loading and preprocessing
2. Degradation generation
3. Multi-model restoration (SwinIR, NAFNet, DiffBIR)
4. Observability map computation
5. Result packaging and storage

Key Features:
- Aggressive GPU memory management with dynamic model loading/unloading
- Progress tracking with tqdm
- Comprehensive error logging
- Batch processing support
- Flexible output formatting

Author: Visual Observability Benchmark Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import logging
import gc
import time
import json
from tqdm import tqdm
import os
import sys

# Import project modules
from ..core.degradation import DegradationEngine, RealWorldDegradationEngine, create_degradation_engine
from ..core.observability_math import generate_observability_map, ObservabilityMetrics
from ..models.wrapper_swinir import SwinIRWrapper
from ..models.wrapper_nafnet import NAFNetWrapper
from ..models.wrapper_diffusion import DiffBIRWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Utility class for loading and preprocessing images.
    """

    def __init__(
        self,
        image_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        to_rgb: bool = True,
        max_size: Optional[int] = 1024
    ):
        """
        Args:
            image_size: Target size (H, W), if None use original
            normalize: Whether to normalize to [-1, 1]
            to_rgb: Convert to RGB
            max_size: Maximum image dimension to prevent OOM (0 means no limit)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.to_rgb = to_rgb
        self.max_size = max_size

        # Default transform
        self.transform_list = []

        if to_rgb:
            self.transform_list.append(transforms.Lambda(lambda x: x.convert('RGB')))

        if image_size is not None:
            self.transform_list.append(transforms.Resize(image_size))

        self.transform_list.extend([
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
        ])

        if normalize:
            self.transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = transforms.Compose(self.transform_list)

    def load_image(self, path: Union[str, Path]) -> torch.Tensor:
        """Load single image from path with optional size limit."""
        img = Image.open(path)

        # Resize if image is too large (prevent OOM)
        if self.max_size and self.max_size > 0:
            w, h = img.size
            if max(w, h) > self.max_size:
                # Calculate new size maintaining aspect ratio
                if w > h:
                    new_w = self.max_size
                    new_h = int(h * self.max_size / w)
                else:
                    new_h = self.max_size
                    new_w = int(w * self.max_size / h)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.debug(f"Resized {path}: {w}x{h} -> {new_w}x{new_h}")

        return self.transform(img).unsqueeze(0)  # Add batch dimension

    def load_images_from_folder(
        self,
        folder: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ) -> List[torch.Tensor]:
        """Load all images from folder."""
        folder = Path(folder)
        image_paths = []

        for ext in extensions:
            image_paths.extend(folder.glob(f'*{ext}'))
            image_paths.extend(folder.glob(f'*{ext.upper()}'))

        images = []
        for path in sorted(image_paths):
            try:
                img = self.load_image(path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        return images


class ImageSaver:
    """
    Utility class for saving images and observability maps.
    """

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
        """Convert tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension

        # Denormalize if needed (from [-1,1] to [0,1])
        if denormalize:
            tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        # Clamp to valid range
        tensor = torch.clamp(tensor, 0.0, 1.0)

        # Convert to numpy
        arr = tensor.cpu().numpy()
        arr = (arr * 255).astype(np.uint8)

        # Handle channels
        if arr.shape[0] == 3:  # RGB
            arr = np.transpose(arr, (1, 2, 0))
            return Image.fromarray(arr, mode='RGB')
        elif arr.shape[0] == 1:  # Grayscale
            arr = arr.squeeze(0)
            return Image.fromarray(arr, mode='L')
        else:
            raise ValueError(f"Unexpected channel dimension: {arr.shape[0]}")

    @staticmethod
    def save_tensor(
        tensor: torch.Tensor,
        path: Union[str, Path],
        denormalize: bool = True,
        create_dirs: bool = True
    ) -> None:
        """Save tensor as image file."""
        path = Path(path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        img = ImageSaver.tensor_to_pil(tensor, denormalize=denormalize)
        img.save(path)

    @staticmethod
    def save_observability_map(
        observability_map: torch.Tensor,
        path: Union[str, Path],
        colormap: str = 'jet',
        create_dirs: bool = True,
        invert_colors: bool = True
    ) -> None:
        """
        Save observability map as grayscale or colormap image.

        Args:
            observability_map: Observability tensor [B, 1, H, W] or [H, W]
            path: Output path
            colormap: Colormap name ('jet', 'viridis', 'gray', 'hot', 'coolwarm')
            create_dirs: Whether to create parent directories
            invert_colors: If True, invert the colormap so that:
                          - Red = Low observability (Danger/Hallucination)
                          - Blue = High observability (Safe)
        """
        path = Path(path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        if observability_map.dim() == 4:
            observability_map = observability_map.squeeze(0)  # [1, H, W] -> [H, W]
        if observability_map.dim() == 3:
            observability_map = observability_map.squeeze(0)  # [1, H, W] -> [H, W]

        # Convert to numpy and scale to [0, 255]
        arr = observability_map.cpu().numpy()
        arr = np.clip(arr, 0, 1)

        # Invert if requested (for JET: Red=Low=Danger)
        if invert_colors and colormap in ['jet', 'viridis', 'coolwarm', 'hot']:
            arr = 1.0 - arr

        # Apply colormap
        import matplotlib.cm as cm
        if colormap == 'gray':
            arr_uint8 = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr_uint8, mode='L')
        else:
            colored = cm.get_cmap(colormap)(arr)[:, :, :3]
            colored = (colored * 255).astype(np.uint8)
            img = Image.fromarray(colored, mode='RGB')

        img.save(path)

    @staticmethod
    def save_metadata(
        metadata: Dict[str, Any],
        path: Union[str, Path]
    ) -> None:
        """Save metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            else:
                return obj

        metadata = convert_types(metadata)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class DatasetGenerator:
    """
    Main dataset generator class.

    Generates Visual Observability Benchmark datasets by:
    1. Loading ground truth images
    2. Generating degraded versions
    3. Running multiple restoration models
    4. Computing observability maps
    5. Saving results

    Memory Management:
    - Models are dynamically loaded/unloaded for each image/batch
    - GPU memory is cleared after each model inference
    - CPU tensors are used for storage to minimize VRAM usage

    Args:
        gt_folder: Folder containing ground truth images
        output_folder: Output folder for generated dataset
        scale: Upscaling factor (default: 4)
        device: Computation device (auto-detect if None)
        batch_size: Number of images to process simultaneously
        degradation_type: Type of degradation ('classical' or 'realworld')
    """

    def __init__(
        self,
        gt_folder: Union[str, Path],
        output_folder: Union[str, Path],
        scale: int = 4,
        device: Optional[str] = None,
        batch_size: int = 1,
        degradation_type: str = 'classical',
        observability_alpha: float = 1.0,
        observability_beta: float = 0.5,
        max_image_size: int = 512
    ):
        """
        Args:
            gt_folder: Folder containing ground truth images
            output_folder: Output folder for generated dataset
            scale: Upscaling factor (default: 4)
            device: Computation device (auto-detect if None)
            batch_size: Number of images to process simultaneously
            degradation_type: Type of degradation ('classical' or 'realworld')
            observability_alpha: Alpha parameter for observability calculation
            observability_beta: Beta parameter for observability calculation
            max_image_size: Maximum image dimension to prevent OOM (default: 512)
                           Set to 0 for no limit (not recommended for large images)
        """
        self.gt_folder = Path(gt_folder)
        self.output_folder = Path(output_folder)
        self.scale = scale
        self.batch_size = batch_size
        self.degradation_type = degradation_type
        self.observability_alpha = observability_alpha
        self.observability_beta = observability_beta
        self.max_image_size = max_image_size

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize components
        self.image_loader = ImageLoader(max_size=max_image_size)
        self.degradation_engine = create_degradation_engine(
            engine_type=degradation_type,
            scale=scale
        )
        self.observability_metrics = ObservabilityMetrics(
            alpha=observability_alpha,
            beta=observability_beta
        )

        # Create output directories
        self._setup_output_folders()

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        logger.info(f"DatasetGenerator initialized:")
        logger.info(f"  GT folder: {self.gt_folder}")
        logger.info(f"  Output folder: {self.output_folder}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Scale: {scale}")
        logger.info(f"  Degradation type: {degradation_type}")

    def _setup_output_folders(self) -> None:
        """Create output folder structure."""
        self.degraded_folder = self.output_folder / 'degraded'
        self.gt_folder_out = self.output_folder / 'ground_truth'
        self.observability_folder = self.output_folder / 'observability'
        self.restored_folder = self.output_folder / 'restored'
        self.metadata_folder = self.output_folder / 'metadata'

        for folder in [self.degraded_folder, self.gt_folder_out,
                       self.observability_folder, self.restored_folder, self.metadata_folder]:
            folder.mkdir(parents=True, exist_ok=True)

    def _get_image_paths(self) -> List[Path]:
        """Get list of image paths from GT folder."""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        paths = []

        for ext in extensions:
            paths.extend(self.gt_folder.glob(f'*{ext}'))
            paths.extend(self.gt_folder.glob(f'*{ext.upper()}'))

        return sorted(paths)

    def _process_single_image(
        self,
        gt_tensor: torch.Tensor,
        image_name: str,
        save_intermediate: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single image through the complete pipeline.

        Memory Management Strategy:
        1. Load GT to GPU
        2. Generate degraded image on GPU
        3. For each model:
           a. Dynamically instantiate model wrapper
           b. Run inference
           c. Collect result to CPU
           d. Delete model instance
           e. Clear CUDA cache
        4. Compute observability map on CPU
        5. Save all results

        Args:
            gt_tensor: Ground truth tensor [1, C, H, W]
            image_name: Name for saving files
            save_intermediate: Whether to save intermediate results

        Returns:
            Dictionary with results and metadata, or None if failed
        """
        try:
            # Aggressive memory cleanup before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # Move GT to device
            gt_tensor = gt_tensor.to(self.device)

            # ==========================================================================
            # Step 1: Generate Degraded Image
            # ==========================================================================
            logger.debug(f"Generating degradation for {image_name}")
            degraded_tensor, deg_metadata = self.degradation_engine.degrade(gt_tensor)

            # Move degraded to CPU for storage
            degraded_cpu = degraded_tensor.cpu().clone()

            # ==========================================================================
            # Step 2: Run Restoration Models (One by One with Memory Cleanup)
            # ==========================================================================
            restored_results = []
            gt_shape = gt_tensor.shape[2:]  # (H, W) of GT

            # Store original size for potential retry
            original_h, original_w = gt_shape

            # --- SwinIR ---
            logger.debug(f"Running SwinIR on {image_name}")
            swinir_success = False
            retry_count = 0
            max_retries = 3

            while not swinir_success and retry_count < max_retries:
                try:
                    # Aggressive memory cleanup before each model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    swinir = SwinIRWrapper(
                        model_type='classical',
                        scale=self.scale,
                        device=self.device
                    )

                    # Load model first to get effective scale
                    swinir._load_model()
                    model_scale = getattr(swinir, '_effective_upscale', self.scale)

                    # Handle model scale mismatch (e.g., x8 model for x4 task)
                    if model_scale != self.scale:
                        # Scale input to match model, then scale output back
                        input_scale_factor = model_scale // self.scale
                        input_h, input_w = gt_shape[0] // input_scale_factor, gt_shape[1] // input_scale_factor
                        degraded_scaled = F.interpolate(
                            degraded_tensor, size=(input_h, input_w),
                            mode='bicubic', align_corners=False
                        )
                        restored_swinir = swinir.restore(degraded_scaled)
                        restored_swinir = F.interpolate(
                            restored_swinir, size=gt_shape, mode='bicubic', align_corners=False
                        )
                    else:
                        restored_swinir = swinir.restore(degraded_tensor)
                        restored_swinir = F.interpolate(
                            restored_swinir, size=gt_shape, mode='bicubic', align_corners=False
                        )

                    restored_results.append(restored_swinir.cpu().clone())
                    del swinir, restored_swinir
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug(f"SwinIR completed for {image_name}")
                    swinir_success = True

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(f"SwinIR OOM for {image_name}, retrying with smaller image ({retry_count}/{max_retries})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        # Halve the image size
                        new_h, new_w = gt_shape[0] // 2, gt_shape[1] // 2
                        gt_tensor = F.interpolate(gt_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        degraded_tensor = F.interpolate(degraded_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        gt_shape = (new_h, new_w)
                        continue
                    else:
                        raise e

            if not swinir_success:
                logger.warning(f"SwinIR failed for {image_name} after {max_retries} retries")
                restored_results.append(torch.zeros(1, 3, gt_shape[0], gt_shape[1]))

            # --- NAFNet ---
            logger.debug(f"Running NAFNet on {image_name}")
            nafnet_success = False
            retry_count = 0

            while not nafnet_success and retry_count < max_retries:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    nafnet = NAFNetWrapper(
                        task='denoise',
                        width=32,  # Match actual GoPro model weights (file name is misleading)
                        n_blocks=8,
                        device=self.device
                    )
                    # For fair comparison, use degraded at same resolution
                    degraded_for_nafnet = F.interpolate(
                        degraded_tensor,
                        size=gt_shape,
                        mode='bicubic',
                        align_corners=False
                    )

                    restored_nafnet = nafnet.restore(degraded_for_nafnet)

                    # Match dimensions to GT
                    restored_nafnet = F.interpolate(restored_nafnet, size=gt_shape, mode='bicubic', align_corners=False)

                    restored_results.append(restored_nafnet.cpu().clone())
                    del nafnet, restored_nafnet
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug(f"NAFNet completed for {image_name}")
                    nafnet_success = True

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(f"NAFNet OOM for {image_name}, retrying with smaller image ({retry_count}/{max_retries})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        # Halve the image size
                        new_h, new_w = gt_shape[0] // 2, gt_shape[1] // 2
                        if new_h < 64 or new_w < 64:
                            raise e  # Can't go smaller
                        gt_tensor = F.interpolate(gt_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        degraded_tensor = F.interpolate(degraded_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        gt_shape = (new_h, new_w)
                        # Update previous results to match new size
                        if len(restored_results) > 0:
                            restored_results[0] = F.interpolate(restored_results[0], size=gt_shape, mode='bicubic', align_corners=False)
                        continue
                    else:
                        raise e

            if not nafnet_success:
                logger.warning(f"NAFNet failed for {image_name} after {max_retries} retries")
                restored_results.append(torch.zeros(1, 3, gt_shape[0], gt_shape[1]))

            # --- DiffBIR ---
            logger.debug(f"Running DiffBIR on {image_name}")
            diffbir_success = False
            retry_count = 0

            while not diffbir_success and retry_count < max_retries:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    diffbir = DiffBIRWrapper(
                        model_type='swinir_ir',
                        stage='restoration',
                        scale=self.scale,
                        device=self.device,
                        diffusion_steps=30  # Reduced for faster processing
                    )
                    restored_diffbir = diffbir.restore(degraded_tensor)
                    # Resize to match GT
                    restored_diffbir = F.interpolate(restored_diffbir, size=gt_shape, mode='bicubic', align_corners=False)
                    restored_results.append(restored_diffbir.cpu().clone())
                    del diffbir, restored_diffbir
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug(f"DiffBIR completed for {image_name}")
                    diffbir_success = True

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(f"DiffBIR OOM for {image_name}, retrying with smaller image ({retry_count}/{max_retries})")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        # Halve the image size
                        new_h, new_w = gt_shape[0] // 2, gt_shape[1] // 2
                        if new_h < 64 or new_w < 64:
                            raise e  # Can't go smaller
                        gt_tensor = F.interpolate(gt_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        degraded_tensor = F.interpolate(degraded_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
                        gt_shape = (new_h, new_w)
                        # Update previous results to match new size
                        for i in range(len(restored_results)):
                            restored_results[i] = F.interpolate(restored_results[i], size=gt_shape, mode='bicubic', align_corners=False)
                        continue
                    else:
                        raise e

            if not diffbir_success:
                logger.warning(f"DiffBIR failed for {image_name} after {max_retries} retries")
                restored_results.append(torch.zeros(1, 3, gt_shape[0], gt_shape[1]))

            # ==========================================================================
            # Step 3: Compute Observability Map (on CPU)
            # ==========================================================================
            logger.debug(f"Computing observability map for {image_name}")

            # Move GT to CPU for computation
            gt_cpu = gt_tensor.cpu()

            observability_map, obs_stats = generate_observability_map(
                gt_cpu,
                restored_results,
                alpha=self.observability_alpha,
                beta=self.observability_beta
            )

            # ==========================================================================
            # Step 4: Save Results
            # ==========================================================================
            base_name = Path(image_name).stem

            # Ensure all tensors are in [0, 1] range for saving
            def to_save_range(t):
                """Convert tensor to [0, 1] range for saving."""
                t = t.clone()
                # If tensor is in [-1, 1] range (from normalize), convert to [0, 1]
                if t.min() < 0:
                    t = t * 0.5 + 0.5
                return torch.clamp(t, 0, 1)

            # Save degraded image
            ImageSaver.save_tensor(
                to_save_range(degraded_cpu),
                self.degraded_folder / f'{base_name}_degraded.png',
                denormalize=False
            )

            # Save GT image
            ImageSaver.save_tensor(
                to_save_range(gt_cpu),
                self.gt_folder_out / f'{base_name}_gt.png',
                denormalize=False
            )

            # Save observability map with JET colormap
            # Red = High Observability (Safe), Blue = Low Observability (Danger)
            ImageSaver.save_observability_map(
                observability_map,
                self.observability_folder / f'{base_name}_observability.png',
                colormap='jet',
                invert_colors=False  # Red = High observability (Safe)
            )

            # Save restored images
            model_names = ['swinir', 'nafnet', 'diffbir']
            for i, (restored, name) in enumerate(zip(restored_results, model_names)):
                ImageSaver.save_tensor(
                    to_save_range(restored),
                    self.restored_folder / f'{base_name}_restored_{name}.png',
                    denormalize=False
                )

            # Save metadata
            metadata = {
                'image_name': image_name,
                'degradation_metadata': deg_metadata,
                'observability_stats': {
                    'mean_error': obs_stats['mean_error'],
                    'mean_variance': obs_stats['mean_variance'],
                    'mean_observability': obs_stats['mean_observability'],
                    'num_models': obs_stats['num_models'],
                    'alpha': obs_stats['alpha'],
                    'beta': obs_stats['beta']
                },
                'pipeline_info': {
                    'scale': self.scale,
                    'degradation_type': self.degradation_type,
                    'device': self.device,
                    'batch_size': self.batch_size
                }
            }

            ImageSaver.save_metadata(
                metadata,
                self.metadata_folder / f'{base_name}_metadata.json'
            )

            # Cleanup GPU tensors
            del gt_tensor, gt_cpu, degraded_tensor, degraded_cpu
            del restored_results
            torch.cuda.empty_cache()
            gc.collect()

            return metadata

        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            self.stats['errors'].append({
                'image': image_name,
                'error': str(e)
            })
            return None

    def process_images(
        self,
        num_images: Optional[int] = None,
        shuffle: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process all images or a subset through the pipeline.

        Args:
            num_images: Number of images to process (None for all)
            shuffle: Whether to shuffle image order
            show_progress: Whether to show tqdm progress bar

        Returns:
            Dictionary with processing statistics
        """
        image_paths = self._get_image_paths()

        if not image_paths:
            logger.warning(f"No images found in {self.gt_folder}")
            return self.stats

        if num_images is not None:
            image_paths = image_paths[:num_images]

        if shuffle:
            import random
            random.shuffle(image_paths)

        logger.info(f"Processing {len(image_paths)} images...")

        # Create progress bar
        progress_bar = tqdm(
            image_paths,
            desc="Generating dataset",
            disable=not show_progress
        )

        for path in progress_bar:
            image_name = path.name

            try:
                # Load image
                gt_tensor = self.image_loader.load_image(path)

                # Process
                result = self._process_single_image(gt_tensor, image_name)

                if result is not None:
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

                self.stats['total_processed'] += 1

                # Update progress bar description
                progress_bar.set_postfix({
                    'success': self.stats['successful'],
                    'failed': self.stats['failed']
                })

            except Exception as e:
                logger.error(f"Failed to load/process {image_name}: {e}")
                self.stats['failed'] += 1
                self.stats['total_processed'] += 1
                self.stats['errors'].append({
                    'image': image_name,
                    'error': str(e)
                })

        # Save final statistics
        self._save_statistics()

        logger.info(f"Processing complete:")
        logger.info(f"  Total: {self.stats['total_processed']}")
        logger.info(f"  Successful: {self.stats['successful']}")
        logger.info(f"  Failed: {self.stats['failed']}")

        return self.stats

    def _save_statistics(self) -> None:
        """Save processing statistics to JSON."""
        stats_path = self.output_folder / 'processing_statistics.json'

        # Convert stats to JSON-safe format
        def convert(obj):
            if isinstance(obj, Exception):
                return str(obj)
            return obj

        stats_json = {
            'summary': {
                k: v if k != 'errors' else len(v)
                for k, v in self.stats.items()
            },
            'errors': self.stats['errors']
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_json, f, indent=2, ensure_ascii=False)


class ObservabilityPipeline:
    """
    Lightweight pipeline for computing observability maps on existing images.

    Use this when you have already generated degraded/restored images
    and just need to compute observability maps.

    Args:
        alpha: Weight for error term in observability calculation
        beta: Weight for variance term
        device: Computation device
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        device: Optional[str] = None
    ):
        self.alpha = alpha
        self.beta = beta

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.metrics = ObservabilityMetrics(alpha=alpha, beta=beta)

    def compute_from_paths(
        self,
        gt_paths: List[Union[str, Path]],
        restored_paths_list: List[List[Union[str, Path]]],
        output_folder: Union[str, Path],
        save_visualization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute observability maps from image paths.

        Args:
            gt_paths: List of ground truth image paths
            restored_paths_list: List of K lists, each containing paths
                                  for images restored by one model
            output_folder: Output folder for results
            save_visualization: Whether to save visualization images

        Returns:
            List of result dictionaries
        """
        output_folder = Path(output_folder)
        obs_folder = output_folder / 'observability'
        obs_folder.mkdir(parents=True, exist_ok=True)

        results = []
        loader = ImageLoader(normalize=False)  # Load without normalization for saving

        for i, gt_path in enumerate(tqdm(gt_paths, desc="Computing observability")):
            try:
                # Load GT
                gt_tensor = loader.load_image(gt_path)

                # Load restored images for each model
                restored_tensors = []
                for model_paths in restored_paths_list:
                    if i < len(model_paths):
                        rest_tensor = loader.load_image(model_paths[i])
                        restored_tensors.append(rest_tensor)

                if len(restored_tensors) < 2:
                    logger.warning(f"Not enough restored images for {gt_path}")
                    continue

                # Compute observability
                obs_map, stats = generate_observability_map(
                    gt_tensor,
                    restored_tensors,
                    alpha=self.alpha,
                    beta=self.beta
                )

                # Save
                base_name = Path(gt_path).stem
                ImageSaver.save_observability_map(
                    obs_map,
                    obs_folder / f'{base_name}_observability.png'
                )

                results.append({
                    'image_name': Path(gt_path).name,
                    'observability': obs_map,
                    'stats': stats
                })

            except Exception as e:
                logger.error(f"Failed to process {gt_path}: {e}")

        return results


# ============================================================================
# Utility Functions
# ============================================================================

def create_dataset_from_folder(
    gt_folder: Union[str, Path],
    output_folder: Union[str, Path],
    scale: int = 4,
    num_images: Optional[int] = None,
    degradation_type: str = 'classical',
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to create a dataset from a folder of images.

    Args:
        gt_folder: Folder containing GT images
        output_folder: Output folder for dataset
        scale: Upscaling factor
        num_images: Number of images to process
        degradation_type: Type of degradation
        device: Computation device

    Returns:
        Processing statistics
    """
    generator = DatasetGenerator(
        gt_folder=gt_folder,
        output_folder=output_folder,
        scale=scale,
        device=device,
        degradation_type=degradation_type
    )

    return generator.process_images(num_images=num_images)


def batch_compute_observability(
    gt_folder: Union[str, Path],
    restored_folders: List[Union[str, Path]],
    output_folder: Union[str, Path],
    alpha: float = 1.0,
    beta: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Compute observability maps for multiple restoration model outputs.

    Args:
        gt_folder: Folder with ground truth images
        restored_folders: List of folders with restored images
        output_folder: Output folder
        alpha: Observability alpha parameter
        beta: Observability beta parameter

    Returns:
        List of results
    """
    from natsort import natsorted

    # Get sorted image paths from each folder
    all_paths = [natsorted(list(Path(f).glob('*.png'))) for f in restored_folders]

    # Get GT paths
    gt_paths = natsorted(list(Path(gt_folder).glob('*.png')))

    # Create paths list for each model
    restored_paths_list = [[p[i] if i < len(p) else None for p in all_paths]
                           for i in range(len(all_paths[0]))]

    # Filter out None paths
    valid_indices = [i for i in range(len(gt_paths))
                     if all(p[i] is not None for p in restored_paths_list)]

    gt_paths = [gt_paths[i] for i in valid_indices]
    restored_paths_list = [[p[i] for p in restored_paths_list] for i in valid_indices]

    # Create pipeline and compute
    pipeline = ObservabilityPipeline(alpha=alpha, beta=beta)
    return pipeline.compute_from_paths(
        gt_paths,
        restored_paths_list,
        output_folder
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example: Generate dataset from a folder of images
    import argparse

    parser = argparse.ArgumentParser(description='Generate Visual Observability Dataset')
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Folder containing ground truth images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder for generated dataset')
    parser.add_argument('--scale', type=int, default=4,
                        help='Upscaling factor (default: 4)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to process')
    parser.add_argument('--deg_type', type=str, default='classical',
                        choices=['classical', 'realworld'],
                        help='Degradation type')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Run dataset generation
    stats = create_dataset_from_folder(
        gt_folder=args.gt_folder,
        output_folder=args.output,
        scale=args.scale,
        num_images=args.num_images,
        degradation_type=args.deg_type,
        device=args.device
    )

    print(f"\nProcessing complete!")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
