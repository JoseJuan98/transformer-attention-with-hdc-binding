# -*- coding: utf-8 -*-
"""Backend Verification Script."""
# Standard imports
import argparse
import sys

# Third party imports
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--exptected-backend",
    "-eb",
    dest="exptected_backend",
    type=str,
    required=True,
    help="Expected backend to use. Options: cpu, cuda, mps, intel, rocm",
)

expected_backend = parser.parse_args().exptected_backend

print(f"\nPyTorch version: {torch.__version__}\n")

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f" => CUDA available: {cuda_available}")
if cuda_available:
    print(f"\tCUDA device count: {torch.cuda.device_count()}")
    print(f"\tCUDA device name: {torch.cuda.get_device_name(0)}\n")

# Check MPS (Apple Silicon)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f" => MPS available: {mps_available}\n")

# Check for Intel extension
try:
    # Third party imports
    import intel_extension_for_pytorch as ipex  # noqa

    print(f" => Intel Extension for PyTorch available: {True}")
    print(f"\tIntel Extension version: {ipex.__version__}")
except ImportError:
    print(f" => Intel Extension for PyTorch available: {False}")

# Determine device
device = "cpu"
if cuda_available:
    device = "cuda"
elif mps_available:
    device = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"

print(f"\n => Using device: {device}")

# Test tensor operations
x = torch.rand(5, 3).to(device)
y = torch.rand(5, 3).to(device)
z = x + y
print(f"\nTest tensor on {device}:")
print(z)

# Verify backend matches the expected backend
actual_backend = "cpu"
if cuda_available:
    actual_backend = "cuda"
elif mps_available:
    actual_backend = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    actual_backend = "intel"

if expected_backend == "intel":
    try:
        # Third party imports
        import intel_extension_for_pytorch  # noqa

        actual_backend = "intel"
    except ImportError:
        pass
elif expected_backend == "rocm" and cuda_available:
    # ROCM shows up as CUDA in PyTorch
    actual_backend = "rocm"

if expected_backend != actual_backend and not (expected_backend == "rocm" and actual_backend == "cuda"):
    print(f"\nERROR: Expected {expected_backend} backend but got {actual_backend}")
    sys.exit(1)
else:
    print(f"\nSUCCESS: Correctly using {actual_backend} backend")
