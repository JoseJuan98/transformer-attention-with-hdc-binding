#!/bin/bash
set -euo pipefail

# ------------------- Configuration -------------------
readonly PYTORCH_VERSION=${PYTORCH_VERSION:-2.6.0}
readonly TRITON_VERSION=${TRITON_VERSION:-3.2.0}
readonly ROCM_VERSION=${ROCM_VERSION:-6.2.4}
readonly INTEL_EXTENSION_VERSION=${INTEL_EXTENSION_VERSION:-2.6.10+xpu}
# Remove the dot from the CUDA version, e.g., 12.4 -> 124
readonly CUDA_VERSION=${CUDA_VERSION:-124}
# Default backend is CPU is not specified
BACKEND=${backend:-cpu}
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


print_message "$BLUE" "\n   PyTorch Project Installation Script"
print_message "$BLUE" "=========================================="
echo -e "\tPYTORCH_VERSION = ${PYTORCH_VERSION} \n\tTRITON_VERSION  = ${TRITON_VERSION}"
echo -e "\tROCM_VERSION    = ${ROCM_VERSION} \nINTEL_EXTENSION_VERSION = ${INTEL_EXTENSION_VERSION}"
echo -e "\tCUDA_VERSION    = ${CUDA_VERSION}"
print_message "$BLUE" "==========================================\n"

# Ask if the user wants to install the dev dependencies


function ask_install_dev() {
  read -p "Are the dependencies above correct? Please notice that only the ones that apply to your backend are going to be used (y/n): " choice
  case "$choice" in
    [Yy]* ) echo "Proceeding ..." ;;
    [Nn]* ) exit 0 ;;
    * ) echo "Please answer yes or no." && ask_install_dev ;;
  esac
}
# if CI=true, skip `ask_install_dev`
if [[ "${CI:-}" == "true" ]]; then
  echo "CI environment detected, skipping user confirmation."
else
  print_message "$YELLOW" "Please review the configuration above before proceeding."
  ask_install_dev
fi



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

# Detect backend based on hardware, unless overridden by command line
if [[ "$BACKEND" == "cpu" ]]; then # Only autodetect if backend is default (cpu)
  # Check for Apple Silicon
  if [ "$OS" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    print_message "$GREEN" "Apple Silicon detected"
    BACKEND="mps"
  fi

  # Check for Intel GPU
  if check_and_print_command "sycl-ls" "Intel GPU with SYCL support detected"; then
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
  if check_and_print_command "nvidia-smi" "NVIDIA GPU detected"; then
    check_and_print_command "nvidia-smi" "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
    BACKEND="cuda"
  elif [ -d "/proc/driver/nvidia" ] || [ -f "/proc/driver/nvidia/version" ]; then
    print_message "$GREEN" "NVIDIA drivers detected"
    BACKEND="cuda"
  fi
fi

DEV=""
if $INSTALL_DEV; then
  print_message "$BLUE" "Installing dev dependencies"
  DEV="--with=dev,lint"
fi

# Determine installation command based on backend
install_pytorch() {
  local WITH_TORCH=""
  local backend="$1"
  print_message "$GREEN" "\n\nInstalling PyTorch with $backend support"
  print_message "$BLUE"  "========================================\n"
  print_message "$GREEN" "Installing Poetry"
  make install-poetry
  print_message "$GREEN" "Generating Poetry lock file for your environment"
  poetry lock --no-cache --regenerate
  print_message "$GREEN" "Installing PyTorch with $backend support"
  local COMMON_DEPS=("lightning>=2.5.0" "torchmetrics>=1.6.2" "torch-tb-profiler>=0.4.3")
  case "$backend" in
    cuda)
      if [[ "$CUDA_VERSION" == 124 ]]; then
        WITH_TORCH="--with torch"
      else
        poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
      fi
      ;;
    rocm)
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-rocm>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}
      ;;
    mps)
#      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/nightly/cpu
      WITH_TORCH="--with torch"
      ;;
    intel)
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" "pytorch-triton-xpu>=${TRITON_VERSION}" --index-url https://download.pytorch.org/whl/xpu
      poetry run pip install -U --force-reinstall --no-cache-dir "intel-extension-for-pytorch==${INTEL_EXTENSION_VERSION}" "oneccl_bind_pt==${PYTORCH_VERSION}+xpu" --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
      ;;
    *) # CPU
      poetry run pip install -U --force-reinstall --no-cache-dir "torch>=${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
      ;;
  esac
  print_message "$GREEN" "Installing project dependencies"
  if [[ "$WITH_TORCH" == "" ]]; then
    poetry run pip install -U --no-cache-dir $COMMON_DEPS
  fi

  poetry install --no-cache $DEV $WITH_TORCH

  if [[ "$DEV" == "--with dev" ]]; then
    make install-precommit
  fi
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

# Run verification script
poetry run python scripts/backend_verification.py -eb $BACKEND

# Check if verification was successful
if [ $? -eq 0 ]; then
  print_message "$GREEN" "\nPyTorch installation verified successfully!"
  print_message "$BLUE" "Installation complete. You can now use your PyTorch project with $BACKEND backend."
else
  print_message "$RED" "\nPyTorch installation verification failed!"
  print_message "$RED" "Please check the error messages above."
  exit 1
fi
