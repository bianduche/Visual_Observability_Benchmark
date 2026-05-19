"""
Configuration for Visual Observability Benchmark
=================================================

Default configuration for dataset generation.
Can be overridden via command line arguments or programmatic API.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Main configuration for the dataset generation pipeline."""

    # Paths
    gt_folder: Path = Path("./images")
    output_folder: Path = Path("./output")

    # Model settings
    scale: int = 4
    device: Optional[str] = None  # None means auto-detect

    # Degradation settings
    degradation_type: str = "classical"  # 'classical' or 'realworld'
    blur_sigma: Optional[float] = None
    noise_sigma: Optional[float] = None

    # Observability parameters
    observability_alpha: float = 1.0
    observability_beta: float = 0.5

    # Processing settings
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: Optional[int] = None

    # Model-specific settings
    swinir_model_type: str = "classical"
    swinir_tile_size: int = 0
    nafnet_width: int = 32
    nafnet_n_blocks: int = 8
    nafnet_task: str = "denoise"
    diffbir_diffusion_steps: int = 50
    diffbir_cfg_scale: float = 1.5

    # Output settings
    save_degraded: bool = True
    save_gt: bool = True
    save_restored: bool = True
    save_observability: bool = True
    save_metadata: bool = True
    observability_colormap: str = "jet"

    # Logging
    log_file: str = "visual_observability_benchmark.log"
    log_level: str = "INFO"
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.scale not in [1, 2, 3, 4]:
            raise ValueError(f"Scale must be 1, 2, 3, or 4, got {self.scale}")

        if self.degradation_type not in ["classical", "realworld"]:
            raise ValueError(
                f"Degradation type must be 'classical' or 'realworld', "
                f"got {self.degradation_type}"
            )

        if self.observability_alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {self.observability_alpha}")

        if self.observability_beta < 0:
            raise ValueError(f"Beta must be non-negative, got {self.observability_beta}")

    @classmethod
    def from_dict(cls, config: dict) -> "PipelineConfig":
        """Create config from dictionary."""
        # Convert string paths to Path objects
        if "gt_folder" in config and isinstance(config["gt_folder"], str):
            config["gt_folder"] = Path(config["gt_folder"])
        if "output_folder" in config and isinstance(config["output_folder"], str):
            config["output_folder"] = Path(config["output_folder"])

        return cls(**config)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        config = self.__dict__.copy()
        # Convert Path objects to strings for serialization
        config["gt_folder"] = str(config["gt_folder"])
        config["output_folder"] = str(config["output_folder"])
        return config


@dataclass
class ModelConfig:
    """Configuration for individual models."""

    # SwinIR
    swinir_model_path: Optional[str] = None
    swinir_model_type: str = "classical"
    swinir_scale: int = 4
    swinir_tile_size: int = 0
    swinir_tile_pad: int = 16

    # NAFNet
    nafnet_model_path: Optional[str] = None
    nafnet_width: int = 32
    nafnet_n_blocks: int = 8
    nafnet_task: str = "denoise"
    nafnet_tile_size: int = 0

    # DiffBIR
    diffbir_model_path: Optional[str] = None
    diffbir_model_type: str = "swinir_ir"
    diffbir_stage: str = "restoration"
    diffbir_diffusion_steps: int = 50
    diffbir_cfg_scale: float = 1.5
    diffbir_tile_size: int = 0


# Default configurations
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()

# Example configurations for different use cases
PRESETS = {
    "light": PipelineConfig(
        scale=2,
        observability_alpha=0.5,
        observability_beta=0.25,
        degradation_type="classical",
    ),
    "standard": PipelineConfig(
        scale=4,
        observability_alpha=1.0,
        observability_beta=0.5,
        degradation_type="classical",
    ),
    "heavy": PipelineConfig(
        scale=4,
        observability_alpha=2.0,
        observability_beta=1.0,
        degradation_type="realworld",
    ),
}


def load_config(path: str) -> PipelineConfig:
    """Load configuration from JSON file."""
    import json

    with open(path, "r") as f:
        config_dict = json.load(f)

    return PipelineConfig.from_dict(config_dict)


def save_config(config: PipelineConfig, path: str) -> None:
    """Save configuration to JSON file."""
    import json

    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
