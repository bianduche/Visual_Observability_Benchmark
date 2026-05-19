"""
Main Entry Point for Visual Observability Benchmark Dataset Generation
======================================================================

This script provides a command-line interface for generating datasets
with observability maps.

Usage Examples:
    # Basic usage
    python main.py --gt_folder ./images --output ./dataset

    # With options
    python main.py --gt_folder ./images --output ./dataset --scale 4 --num_images 100

    # Real-world degradation
    python main.py --gt_folder ./images --output ./dataset --deg_type realworld

Author: Visual Observability Benchmark Team
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from .pipelines.dataset_generator import (
    DatasetGenerator,
    ObservabilityPipeline,
    create_dataset_from_folder
)
from .core import DegradationEngine, generate_observability_map
from .models import SwinIRWrapper, NAFNetWrapper, DiffBIRWrapper


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('visual_observability_benchmark.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_environment() -> dict:
    """Check and report the environment configuration."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info[f'gpu_{i}'] = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info[f'gpu_{i}_memory_gb'] = round(mem_total, 2)

    return info


def generate_dataset(args) -> None:
    """Generate dataset with observability maps."""
    logging.info("=" * 60)
    logging.info("Visual Observability Benchmark - Dataset Generation")
    logging.info("=" * 60)

    # Check environment
    env_info = check_environment()
    logging.info(f"Environment: {env_info}")

    # Validate paths
    gt_path = Path(args.gt_folder)
    if not gt_path.exists():
        logging.error(f"Ground truth folder does not exist: {gt_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create generator
    logging.info(f"Initializing dataset generator...")
    logging.info(f"  Ground truth folder: {gt_path}")
    logging.info(f"  Output folder: {output_path}")
    logging.info(f"  Scale factor: {args.scale}")
    logging.info(f"  Degradation type: {args.deg_type}")
    logging.info(f"  Device: {args.device or 'auto'}")

    generator = DatasetGenerator(
        gt_folder=gt_path,
        output_folder=output_path,
        scale=args.scale,
        device=args.device,
        degradation_type=args.deg_type,
        batch_size=args.batch_size,
        observability_alpha=args.alpha,
        observability_beta=args.beta,
        max_image_size=args.max_size
    )

    # Process images
    logging.info(f"Starting processing...")
    stats = generator.process_images(
        num_images=args.num_images,
        shuffle=args.shuffle,
        show_progress=not args.no_progress
    )

    # Report results
    logging.info("=" * 60)
    logging.info("Processing Complete!")
    logging.info(f"  Total processed: {stats['total_processed']}")
    logging.info(f"  Successful: {stats['successful']}")
    logging.info(f"  Failed: {stats['failed']}")

    if stats['errors']:
        logging.warning(f"  Errors occurred: {len(stats['errors'])}")
        for err in stats['errors'][:5]:  # Show first 5 errors
            logging.warning(f"    - {err['image']}: {err['error']}")

    logging.info(f"Results saved to: {output_path}")


def compute_observability(args) -> None:
    """Compute observability maps from existing images."""
    logging.info("=" * 60)
    logging.info("Visual Observability Benchmark - Observability Computation")
    logging.info("=" * 60)

    # Validate paths
    gt_path = Path(args.gt_folder)
    restored_folders = [Path(f) for f in args.restored_folders]

    for f in [gt_path] + restored_folders:
        if not f.exists():
            logging.error(f"Path does not exist: {f}")
            sys.exit(1)

    # Create pipeline
    pipeline = ObservabilityPipeline(
        alpha=args.alpha,
        beta=args.beta,
        device=args.device
    )

    # Get image paths
    from natsort import natsorted

    gt_paths = natsorted(list(gt_path.glob('*.png')))
    restored_paths = [natsorted(list(f.glob('*.png'))) for f in restored_folders]

    logging.info(f"Found {len(gt_paths)} ground truth images")
    logging.info(f"Found {len(restored_paths[0])} restored images per model")
    logging.info(f"Using {len(restored_folders)} restoration models")

    # Compute
    results = pipeline.compute_from_paths(
        gt_paths=gt_paths,
        restored_paths_list=restored_paths,
        output_folder=Path(args.output),
        save_visualization=True
    )

    logging.info(f"Computed {len(results)} observability maps")


def test_models(args) -> None:
    """Test individual models."""
    import torch

    logging.info("=" * 60)
    logging.info("Testing Model Wrappers")
    logging.info("=" * 60)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create test input
    test_input = torch.rand(1, 3, 128, 128).to(device)

    models_to_test = {
        'SwinIR': SwinIRWrapper,
        'NAFNet': NAFNetWrapper,
        'DiffBIR': DiffBIRWrapper,
    }

    for name, ModelClass in models_to_test.items():
        logging.info(f"\nTesting {name}...")

        try:
            if name == 'SwinIR':
                model = ModelClass(model_type='classical', scale=4, device=device)
            elif name == 'NAFNet':
                model = ModelClass(task='denoise', device=device)
            else:
                model = ModelClass(device=device)

            result = model.restore(test_input)
            logging.info(f"  Input shape: {test_input.shape}")
            logging.info(f"  Output shape: {result.shape}")
            logging.info(f"  {name} test: PASSED")

            model.unload()
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"  {name} test: FAILED")
            logging.error(f"  Error: {e}")

    logging.info("\nModel testing complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visual Observability Benchmark - Dataset Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset from folder
  python main.py generate --gt_folder ./images --output ./dataset

  # Generate with specific options
  python main.py generate --gt_folder ./images --output ./dataset --scale 2 --num_images 50

  # Compute observability from existing restored images
  python main.py compute --gt_folder ./gt --restored_folders ./restored_swinir ./restored_nafnet --output ./observability

  # Test model wrappers
  python main.py test_models
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate dataset')
    gen_parser.add_argument('--gt_folder', type=str, required=True,
                            help='Folder containing ground truth images')
    gen_parser.add_argument('--output', type=str, required=True,
                            help='Output folder for generated dataset')
    gen_parser.add_argument('--scale', type=int, default=4,
                            help='Upscaling factor (default: 4)')
    gen_parser.add_argument('--num_images', type=int, default=None,
                            help='Number of images to process (default: all)')
    gen_parser.add_argument('--deg_type', type=str, default='classical',
                            choices=['classical', 'realworld'],
                            help='Degradation type (default: classical)')
    gen_parser.add_argument('--device', type=str, default=None,
                            help='Device (cuda/cpu, default: auto)')
    gen_parser.add_argument('--batch_size', type=int, default=1,
                            help='Batch size (default: 1)')
    gen_parser.add_argument('--alpha', type=float, default=1.0,
                            help='Observability alpha (default: 1.0)')
    gen_parser.add_argument('--beta', type=float, default=0.5,
                            help='Observability beta (default: 0.5)')
    gen_parser.add_argument('--shuffle', action='store_true',
                            help='Shuffle image processing order')
    gen_parser.add_argument('--no_progress', action='store_true',
                            help='Hide progress bar')
    gen_parser.add_argument('--max_size', type=int, default=512,
                            help='Maximum image dimension to prevent OOM (default: 512). Set to 0 for no limit.')

    # Compute command
    comp_parser = subparsers.add_parser('compute', help='Compute observability maps')
    comp_parser.add_argument('--gt_folder', type=str, required=True,
                             help='Folder containing ground truth images')
    comp_parser.add_argument('--restored_folders', type=str, nargs='+', required=True,
                             help='Folders containing restored images for each model')
    comp_parser.add_argument('--output', type=str, required=True,
                             help='Output folder for observability maps')
    comp_parser.add_argument('--device', type=str, default=None,
                             help='Device (cuda/cpu, default: auto)')
    comp_parser.add_argument('--alpha', type=float, default=1.0,
                             help='Observability alpha (default: 1.0)')
    comp_parser.add_argument('--beta', type=float, default=0.5,
                             help='Observability beta (default: 0.5)')

    # Test command
    test_parser = subparsers.add_parser('test_models', help='Test model wrappers')
    test_parser.add_argument('--device', type=str, default=None,
                             help='Device (cuda/cpu, default: auto)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Execute command
    if args.command == 'generate':
        generate_dataset(args)
    elif args.command == 'compute':
        compute_observability(args)
    elif args.command == 'test_models':
        test_models(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
