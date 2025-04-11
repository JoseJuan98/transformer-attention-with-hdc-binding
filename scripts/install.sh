#!/bin/bash
set -euo pipefail

# ------------------- Configuration -------------------
readonly PYTORCH_VERSION=2.6.0
readonly TRITON_VERSION=3.2.0
readonly ROCM_VERSION=6.2.4
readonly INTEL_EXTENSION_VERSION=2.6.10+xpu
# Remove the dot from the CUDA version, e.g., 12.4 -> 124
readonly CUDA_VERSION=124

# ------------------- End Configuration -------------------

# ANSI color codes
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Helper function for printing messages
function print_message() {
  local color="$1"
  local message="$2"
  echo -e "${color}${message}${NC}"
}

print_message "$BLUE" "PyTorch Project Installation Script"
print_message "$BLUE" "====================================\n"

# Parse command line arguments
INSTALL_DEV=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      INSTALL_DEV=true
      shift
      ;;
    *)
      print_message "$YELLOW" "Unknown option: $1"
      shift
      ;;
  esac
done

# Check if make is installed
if ! command -v make &> /dev/null; then
  print_message "$RED" "Error: 'make' command not found. Please install make."
  exit 1
fi

print_message "$BLUE" "Detecting hardware and OS..."

# Detect OS
OS=$(uname -s)
print_message "$GREEN" "Operating System: $OS"

# Initialize variables
BACKEND="cpu"  # Default backend

# Helper function to check for a command and print a message
function check_and_print_command() {
  local command="$1"
  local message="$2"
  if command -v "$command" &> /dev/null; then
    print_message "$GREEN" "$message"
    return 0
  else
    return 1
  fi
}

# Check for Apple Silicon
if [ "$OS" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  print_message "$GREEN" "Apple Silicon detected"
  BACKEND="mps"
fi

# Check for Intel GPU
if check_and_print_command "sycl-ls" "Intel GPU with SYCL support detected"; then
  BACKEND="intel"
elif lspci 2>/dev/null | grep -i intel | grep -i vga &> /dev/null; then
  print_message "$GREEN" "Intel GPU detected"
  BACKEND="intel"
fi

# Check for ROCm (AMD GPU)
if check_and_print_command "rocminfo" "ROCm detected"; then
  BACKEND="rocm"
elif [ -d "/opt/rocm" ]; then
  print_message "$GREEN" "ROCm installation detected"
  BACKEND="rocm"
fi

# Check for CUDA
if check_and_print_command "nvidia-smi" "NVIDIA GPU detected: $(nvidia-smi -L | head -n 1)"; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  BACKEND="cuda"
elif [ -d "/proc/driver/nvidia" ] || [ -f "/proc/driver/nvidia/version" ]; then
  print_message "$GREEN" "NVIDIA drivers detected"
  BACKEND="cuda"
fi

if $INSTALL_DEV; then
  print_message "$BLUE" "Installing dev dependencies"
  DEV="--with=dev"
else
  DEV=""
fi

# Determine installation command based on backend
install_pytorch() {
  local backend="$1"
  print_message "$GREEN" "Installing PyTorch with $backend support"
  make install-poetry
  case "$backend" in
    cuda)
#      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
      echo -e "\n"
      ;;
    rocm)
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-rocm>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}
      ;;
    mps)
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cpu
      ;;
    intel)
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-xpu>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/xpu
      poetry run pip install -U --force-reinstall --no-cache-dir "intel-extension-for-pytorch==${INTEL_EXTENSION_VERSION}" "oneccl_bind_pt==${PYTORCH_VERSION}+xpu" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
      ;;
    *) # CPU
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
      ;;
  esac
  poetry install --no-cache $DEV
  make install-precommit
}

# Install PyTorch based on detected backend
case "$BACKEND" in
  cuda|rocm|mps|intel)
    install_pytorch "$BACKEND"
    ;;
  *) # CPU
    print_message "$YELLOW" "No GPU detected. Installing PyTorch with CPU support"
    install_pytorch "cpu"
    ;;
esac

# Verify installation
print_message "$BLUE" "\nVerifying PyTorch installation..."

VERIFICATION_SCRIPT=$(cat <<EOF
import torch
import sys

print(f"PyTorch version: {torch.__version__}")

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Check MPS (Apple Silicon)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")

# Check for Intel extension
try:
    import intel_extension_for_pytorch as ipex
    print(f"Intel Extension for PyTorch available: {True}")
    print(f"Intel Extension version: {ipex.__version__}")
except ImportError:
    print(f"Intel Extension for PyTorch available: {False}")

# Determine device
device = "cpu"
if cuda_available:
    device = "cuda"
elif mps_available:
    device = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"

print(f"Using device: {device}")

# Test tensor operations
x = torch.rand(5, 3).to(device)
y = torch.rand(5, 3).to(device)
z = x + y
print(f"Test tensor on {device}:")
print(z)

# Verify backend matches expected
expected_backend = "$BACKEND"
actual_backend = "cpu"
if cuda_available:
    actual_backend = "cuda"
elif mps_available:
    actual_backend = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    actual_backend = "intel"

if expected_backend == "intel":
    try:
        import intel_extension_for_pytorch
        actual_backend = "intel"
    except ImportError:
        pass
elif expected_backend == "rocm" and cuda_available:
    # ROCm shows up as CUDA in PyTorch
    actual_backend = "rocm"

if expected_backend != actual_backend and not (expected_backend == "rocm" and actual_backend == "cuda"):
    print(f"ERROR: Expected {expected_backend} backend but got {actual_backend}")
    sys.exit(1)
else:
    print(f"SUCCESS: Correctly using {actual_backend} backend")
EOF
)

# Run verification script
poetry run python -c "$VERIFICATION_SCRIPT"

# Check if verification was successful
if [ $? -eq 0 ]; then
  print_message "$GREEN" "\nPyTorch installation verified successfully!"
  print_message "$BLUE" "Installation complete. You can now use your PyTorch project with $BACKEND backend."
else
  print_message "$RED" "\nPyTorch installation verification failed!"
  print_message "$RED" "Please check the error messages above."
  exit 1
fi
