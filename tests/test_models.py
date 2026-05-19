"""
Test script for Visual Observability Benchmark models.
Tests SwinIR, NAFNet, and DiffBIR with a simple image.
"""
import sys
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(f"Project root: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# 1.  Test helper
# ─────────────────────────────────────────────
def tensor_to_pil(tensor):
    """torch.Tensor [0,1] C×H×W → PIL Image"""
    t = tensor.detach().cpu().clamp(0, 1).squeeze(0)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(img):
    """PIL Image → torch.Tensor [0,1] 1×C×H×W"""
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────
# 2.  Create a simple test image
# ─────────────────────────────────────────────
print("\n=== Creating test image ===")
test_img = Image.new("RGB", (256, 256), color=(120, 80, 200))
# Add some gradient/texture
import numpy as np
arr = np.array(test_img).astype(np.float32)
for i in range(256):
    arr[i, :, 0] = int(80 + i * 0.3)
    arr[i, :, 1] = int(120 - i * 0.2)
test_img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
test_img.save(PROJECT_ROOT / "test_input.png")

test_tensor = pil_to_tensor(test_img)
print(f"Input shape: {test_tensor.shape}, range [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")


# ─────────────────────────────────────────────
# 3.  Test NAFNet
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("=== Testing NAFNet ===")
print("="*60)

# Find weight
nafnet_weight = PROJECT_ROOT / "models" / "nafnet" / "NAFNet-GoPro-width32.pth"
print(f"Weight file: {nafnet_weight}  exists={nafnet_weight.exists()}")

if not nafnet_weight.exists():
    # Try project root
    nafnet_weight_alt = PROJECT_ROOT / "NAFNet-GoPro-width32.pth"
    if nafnet_weight_alt.exists():
        nafnet_weight = nafnet_weight_alt

print(f"Using weight: {nafnet_weight}")

try:
    from Visual_Observability_Benchmark.models.wrapper_nafnet import NAFNetWrapper

    print("\n--- Creating NAFNetWrapper ---")
    nafnet = NAFNetWrapper(
        model_path=str(nafnet_weight),
        task='denoise',
        width=32,
        n_blocks=8,
    )

    print("\n--- Loading model ---")
    nafnet._load_model()

    # Check if model is loaded
    if nafnet.model is None:
        print("ERROR: model is None after _load_model()")
    else:
        print(f"Model type: {type(nafnet.model)}")
        total_params = sum(p.numel() for p in nafnet.model.parameters())
        print(f"Total parameters: {total_params:,}")

    print("\n--- Running inference ---")
    result = nafnet.restore(test_tensor)
    print(f"Output shape: {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")

    # Save
    out_path = PROJECT_ROOT / "test_output_nafnet.png"
    tensor_to_pil(result).save(out_path)
    print(f"Saved → {out_path}")

    nafnet.unload()

except Exception as e:
    import traceback
    print(f"NAFNet FAILED: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────
# 4.  Test SwinIR
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("=== Testing SwinIR ===")
print("="*60)

# Find weight
swinir_weight = PROJECT_ROOT / "models" / "swinir" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth"
print(f"Weight file: {swinir_weight}  exists={swinir_weight.exists()}")

if not swinir_weight.exists():
    # Try SwinIR-main/pretrained_models
    for candidate in [
        PROJECT_ROOT / "SwinIR-main" / "experiments" / "pretrained_models" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
        PROJECT_ROOT / "SwinIR-main" / "SwinIR-main" / "experiments" / "pretrained_models" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
    ]:
        if candidate.exists():
            swinir_weight = candidate
            break

print(f"Using weight: {swinir_weight}")

try:
    from Visual_Observability_Benchmark.models.wrapper_swinir import SwinIRWrapper

    print("\n--- Creating SwinIRWrapper ---")
    swinir = SwinIRWrapper(
        model_path=str(swinir_weight),
        model_type='classical',
        scale=8,
    )

    print("\n--- Loading model ---")
    swinir._load_model()

    if swinir.model is None:
        print("ERROR: model is None after _load_model()")
    else:
        print(f"Model type: {type(swinir.model)}")
        total_params = sum(p.numel() for p in swinir.model.parameters())
        print(f"Total parameters: {total_params:,}")
        # Show upsampler
        print(f"upsampler attr: {swinir.model.upsampler}")
        print(f"upscale attr: {swinir.model.upscale}")
        print(f"img_range attr: {swinir.model.img_range}")

    print("\n--- Running inference ---")
    result = swinir.restore(test_tensor)
    print(f"Output shape: {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")

    out_path = PROJECT_ROOT / "test_output_swinir.png"
    tensor_to_pil(result).save(out_path)
    print(f"Saved → {out_path}")

    swinir.unload()

except Exception as e:
    import traceback
    print(f"SwinIR FAILED: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────
# 5.  Test DiffBIR
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("=== Testing DiffBIR ===")
print("="*60)

try:
    from Visual_Observability_Benchmark.models.wrapper_diffusion import DiffBIRWrapper

    print("\n--- Creating DiffBIRWrapper ---")
    diffbir = DiffBIRWrapper(
        model_type='swinir_ir',
        stage='restoration',
        scale=1,
    )

    print("\n--- Loading model ---")
    diffbir._load_model()

    if diffbir.model is None:
        if getattr(diffbir, '_use_simplified', False):
            print("Model: using simplified diffusion implementation (DiffBIR diffusors not available)")
        else:
            print("ERROR: model is None after _load_model()")
    else:
        print(f"Model type: {type(diffbir.model)}")

    print("\n--- Running inference ---")
    result = diffbir.restore(test_tensor)
    print(f"Output shape: {result.shape}, range [{result.min():.3f}, {result.max():.3f}]")

    out_path = PROJECT_ROOT / "test_output_diffbir.png"
    tensor_to_pil(result).save(out_path)
    print(f"Saved → {out_path}")

    diffbir.unload()

except Exception as e:
    import traceback
    print(f"DiffBIR FAILED: {e}")
    traceback.print_exc()


# ─────────────────────────────────────────────
# 6.  Summary
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("Done. Check the output images:")
print(f"  {PROJECT_ROOT / 'test_output_nafnet.png'}")
print(f"  {PROJECT_ROOT / 'test_output_swinir.png'}")
print(f"  {PROJECT_ROOT / 'test_output_diffbir.png'}")
print("They should look like a restored (denoised/deblurred) version of test_input.png")
