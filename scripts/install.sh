#!/bin/bash
set -e

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}PyTorch Project Installation Script${NC}"
echo -e "${BLUE}====================================${NC}\n"

# Parse command line arguments
INSTALL_DEV=false
for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Check if make is installed
if ! command -v make &> /dev/null; then
    echo -e "${RED}Error: 'make' command not found. Please install make.${NC}"
    exit 1
fi

echo -e "${BLUE}Detecting hardware and OS...${NC}"

# Detect OS
OS="$(uname -s)"
echo -e "Operating System: ${GREEN}$OS${NC}"

# Initialize variables
HAS_CUDA=false
HAS_ROCM=false
HAS_MPS=false
HAS_INTEL=false
BACKEND="cpu"  # Default backend

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "NVIDIA GPU detected: ${GREEN}$(nvidia-smi -L | head -n 1)${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    HAS_CUDA=true
    BACKEND="cuda"
elif [ -d "/proc/driver/nvidia" ] || [ -f "/proc/driver/nvidia/version" ]; then
    echo -e "${GREEN}NVIDIA drivers detected${NC}"
    HAS_CUDA=true
    BACKEND="cuda"
fi

# Check for ROCm (AMD GPU)
if command -v rocminfo &> /dev/null; then
    echo -e "${GREEN}ROCm detected${NC}"
    HAS_ROCM=true
    BACKEND="rocm"
elif [ -d "/opt/rocm" ]; then
    echo -e "${GREEN}ROCm installation detected${NC}"
    HAS_ROCM=true
    BACKEND="rocm"
fi

# Check for Apple Silicon
if [ "$OS" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo -e "${GREEN}Apple Silicon detected${NC}"
    HAS_MPS=true
    BACKEND="mps"
fi

# Check for Intel GPU
if command -v sycl-ls &> /dev/null; then
    echo -e "${GREEN}Intel GPU with SYCL support detected${NC}"
    HAS_INTEL=true
    BACKEND="intel"
elif lspci 2>/dev/null | grep -i intel | grep -i vga &> /dev/null; then
    echo -e "${GREEN}Intel GPU detected${NC}"
    HAS_INTEL=true
    BACKEND="intel"
fi

DEV=""
if [ "$INSTALL_DEV" = true ]; then
    DEV="--with=dev"
    echo -e "${BLUE}Installing dev dependencies${NC}"
fi

PYTORCH_VERSION=2.6.0
TRITON_VERSION=3.2.0
ROCM_VERSION=6.2.4
INTEL_EXTENSION_VERSION=2.6.10+xpu
CUDA_VERSION=124

# Priority: CUDA > ROCm > MPS > Intel > CPU
if [ "$HAS_CUDA" = true ]; then
    echo -e "${GREEN}Installing PyTorch with CUDA support${NC}"
    BACKEND="cuda"
    make install-poetry &&
    # TODO: remove CUDA_VERSION dot to get cu124
    poetry install --no-cache $DEV &&
#    poetry run pip install -U --force --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION} &&
    make install-precommit
elif [ "$HAS_ROCM" = true ]; then
    echo -e "${GREEN}Installing PyTorch with ROCm support${NC}"
    BACKEND="rocm"
    make install-poetry &&
    poetry run pip install -U --force --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-rocm>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION} &&
    poetry install --no-cache $DEV &&
    make install-precommit
elif [ "$HAS_MPS" = true ]; then
    echo -e "${GREEN}Installing PyTorch with MPS support${NC}"
    BACKEND="mps"
    make install-poetry &&
    poetry run pip install -U --force --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cpu &&
    poetry install --no-cache $DEV &&
    make install-precommit
elif [ "$HAS_INTEL" = true ]; then
    echo -e "${GREEN}Installing PyTorch with Intel GPU support${NC}"
    BACKEND="intel"
    make install-poetry &&
    poetry run pip install -U --force --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-xpu>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/xpu &&
    poetry run pip install -U --force --no-cache-dir "intel-extension-for-pytorch==${INTEL_EXTENSION_VERSION}" "oneccl_bind_pt==${PYTORCH_VERSION}+xpu" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ &&
    poetry install --no-cache $DEV &&
    make install-precommit
else
    echo -e "${YELLOW}No GPU detected. Installing PyTorch with CPU support${NC}"
    BACKEND="cpu"
    make install-poetry &&
    poetry run pip install -U --force --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu &&
    poetry install --no-cache $DEV &&
    make install-precommit
fi

# Verify installation
echo -e "\n${BLUE}Verifying PyTorch installation...${NC}"

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
    echo -e "\n${GREEN}PyTorch installation verified successfully!${NC}"
    echo -e "${BLUE}Installation complete. You can now use your PyTorch project with $BACKEND backend.${NC}"
else
    echo -e "\n${RED}PyTorch installation verification failed!${NC}"
    echo -e "${RED}Please check the error messages above.${NC}"
    exit 1
fi
