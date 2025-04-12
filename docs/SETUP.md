
# Setup Guide

This guide outlines the steps to set up the development environment for this project.

## Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.11 or higher:**  The project requires Python 3.11 or a later version. Download the appropriate installer from [python.org](https://www.python.org/downloads/). Or use any tool of your choosing, e.g. `poetry` now supports installing python runtimes.
*   **A Python Environment Manager:**  You'll need a tool to manage your Python environment and dependencies.  We recommend **Poetry** or **pip** with `virtualenv`, but other tools like `conda`, `uv`, `pipx` are also compatible.
*   **Make (Optional, but Recommended):**  The project includes a `Makefile` to automate common tasks like dependency installation and testing.  While optional, using `make` simplifies the setup process.

## Hardware Acceleration Support

This project provides comprehensive support for multiple hardware acceleration backends to optimize performance across different systems.

The installation script automatically detects your hardware and installs the appropriate backend. And meanwhile training if the experiment configuration
attribute accelerator is set to `auto` it will automatically select the best available hardware accelerator. This ensures that you can leverage the full potential of your hardware without manual configuration.

If the backend is installed properly, and meanwhile training is not detected automatically, you can set the accelerator attribute in the experiment configuration to `cpu`, `gpu`, `tpu`, `hpu` or `mps` to use the specific backend.

- Choose `gpu` for NVIDIA CUDA, AMD ROCM or Intel discrete GPUs.
- Choose `tpu` for TPU devices.
- Choose `hpu` for Gaudi HPUs.
- Choose `mps` for Apple Silicon GPUs.

For more information on Lightning hardware acceleration with ROCM check [here](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-lightning/README.html)
For more information on Lightning hardware acceleration with Intel HPU check [here](https://www.intel.com/content/www/us/en/developer/articles/training/introduction-to-pytorch-lightning.html)

The supported backends include:

### Supported Platforms

Custom version of PyTorch can be specified in the `make install` command along any other platform specific version of
dependencies by defining the `PYTORCH_VERSION`, e.g. `make install PYTORCH_VERSION=2.8.0`.

*   **NVIDIA CUDA** - Full support for NVIDIA GPUs
    *   Default: CUDA 12.4
    *   Custom versions can be specified in the `make install` command by defining the `CUDA_VERSION` variable without dots,
e.g. for version 12.6 -> `make install CUDA_VERSION=126`

*   **AMD ROCm** - Optimized for AMD GPUs
    *   Default: ROCm 6.2.4
    *   Custom versions can be specified in the `make install` command by defining the `ROCM_VERSION` and `TRITON_VERSION` variables,
e.g. `make install ROCM_VERSION=6.4.0 TRITON_VERSION=3.2.0`, remember to specify the `TRITON_VERSION` (`pytorch-triton-rocm`)
corresponding to the ROCM version you are using.

*   **Apple Silicon (MPS)** - Native acceleration for Apple M1/M2/M3 chips
    *   Leverages Metal Performance Shaders for optimal performance
    *   Only the PyTorch version can be specified

*   **Intel GPU** - Support for Intel GPUs with SYCL
        *   Custom versions can be specified in the `make install` command by defining the `INTEL_EXTENSION_VERSION` and `TRITON_VERSION` variables,
e.g. `make install INTEL_EXTENSION_VERSION=2.6.10+xpu TRITON_VERSION=3.2.0`, remember to specify the `TRITON_VERSION` (`pytorch-triton-xpu`)
corresponding to the `intel-extension-for-pytorch` version you are using.

*   **CPU-only** - Reliable fallback option
    *   Ensures compatibility on systems without supported GPUs or when troubleshooting hardware-specific issues
    *   Only the PyTorch version can be specified

### Automatic Configuration

The `make install` command intelligently detects your hardware configuration and installs the appropriate acceleration backend without requiring manual intervention.  The `Makefile` handles the backend selection automatically.

## Installing Make (Recommended)

`make` is a build automation tool. If you choose to use it, follow the instructions below for your operating system:

*   **Windows:**
    1.  Open PowerShell as an administrator.
    2.  Run: `winget install GnuWin32.Make`
    3.  Add the installation directory (typically `C:\Program Files (x86)\GnuWin32\bin`) to your system's `PATH` environment variable.  You can do this by searching for "environment variables" in the Windows search bar, clicking "Edit the system environment variables," then "Environment Variables...", selecting "Path" under "System variables," clicking "Edit," and adding the path.
*   **macOS:**
    1.  Open a terminal.
    2.  Run: `brew install make` (This requires [Homebrew](https://brew.sh/) to be installed).
*   **Linux:** `make` is usually pre-installed on most Linux distributions.  You can verify by running `make --version` in your terminal. If it's not installed, use your distribution's package manager (e.g., `apt-get install make` on Debian/Ubuntu, `yum install make` on Fedora/CentOS).

## Setting Up the Python Environment

### Recommended Method: Using Make

The project provides simple make commands that handle the entire installation process, including hardware detection and appropriate PyTorch backend installation:


1. **For standard installation:** create a virtual environment (**recommended**) with any tool of your choice.
    * The python path is usually `/usr/bin/python3.XY` on Linux and macOS, or `C:\Program Files\Python3.XY\python.exe` on Windows.
    * Poetry: install poetry and the python version you need `make install-poetry && poetry python install 3.12`
    * Virtualenv: `python -m virtualenv .venv --python=<path to your python version>`


2. **Install dependencies:**

    ```bash
    make install
    ```

    This command will:
    *   Install Poetry if not already installed
    *   Detect your hardware (NVIDIA CUDA, AMD ROCm, Apple Silicon, Intel GPU, or CPU-only)
    *   Install the appropriate PyTorch backend for your hardware
    *   Set up the project environment

3. **For development setup:**

    ```bash
    make install-dev
    ```

    This command does everything the standard installation does, plus:
    *   Installs development dependencies
    *   Sets up pre-commit hooks
    *   Installs testing tools

4. **Run experiments:**

    After installation, you can run experiments using the provided scripts. For example, to run a time series experiment:

    ```bash
    make run-ts
    ```

### Alternative Methods

If you cannot use the make commands for some reason, you can use these alternative approaches:

#### Using the Installation Script Directly

```bash
chmod +x scripts/install.sh  # Make the script executable (Linux/macOS only)
./scripts/install.sh         # Run the standard installation
```

or for development setup:

```bash
./scripts/install.sh --dev   # Install with development dependencies
```

#### Using Poetry Manually (Less Recommended)

While the `Makefile` automates backend selection, you *can* still use Poetry directly, but it's less convenient.  The `Makefile` is the preferred method.

1. **Install Poetry:**

    ```bash
    pip install --no-cache-dir -U "poetry>=2.1.1"
    poetry config virtualenvs.in-project true --local
    ```

2. **Install project dependencies**:

    ```bash
    poetry install --no-cache
    ```

3. **For development dependencies**:

    ```bash
    poetry install --no-cache --with=dev
    poetry run pre-commit install
    ```

4. **Install hardware especific dependencies** (The Makefile handles this automatically):

   ```bash
    # The Makefile handles this automatically. Do not use these unless you have a specific reason.
    poetry run pip install -U --force-reinstall --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu126
    ```

## Verifying Your Installation

To verify that PyTorch is correctly installed with the appropriate backend:

```bash
poetry run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch, 'mps') and torch.mps.is_available() else 'cpu'); print(f'Using device: {device}'); x = torch.rand(5, 3).to(device); print(x)"
```

This command will show your PyTorch version, hardware acceleration availability, and run a simple tensor operation on your detected device.

## Setting Up Automatic Formatting

To automatically format your code according to the project's style (Black), follow these steps:

1.  **Install Development Dependencies:** Ensure you've installed the development dependencies using `make install-dev`. This will install the `black` formatter.

### In PyCharm

2. **Configure PyCharm:**
    *   Open PyCharm settings (File > Settings or Ctrl+Alt+S).
    *   Navigate to "Tools" > "Black".
    *   Check "On code reformat" and "On save".
    *   Set the settings to `-l 120`

![](attachments/black_on_save.png)

### In VSCode

2.  **Configure VS Code:**
    *   Open VS Code settings (File > Preferences > Settings or Ctrl+,).
    *   Search for "Format on Save".
    *   Enable "Editor: Format On Save".
    *   Search for "Python > Formatting: Provider".
    *   Select "black" from the dropdown.
    *   (Optional, but recommended) Search for "Editor: Format On Save Mode" and set it to "modifications" if you only want to format changed lines, or "file" to format the entire file.

## Setting Up Docstring Style (PyCharm)

To ensure consistent docstring formatting (Google style) in PyCharm:

1.  Open PyCharm settings (File > Settings or Ctrl+Alt+S).
2.  Navigate to "Tools" > "Python Integrated Tools".
3.  Under "Docstrings," set "Docstring format" to "Google".

![](attachments/change_docstring_style.png)

## Pre-commit Hooks

When you run `make install-dev`, pre-commit hooks are automatically installed. These hooks run checks before each commit to ensure code quality.

To manually run the pre-commit checks:

```bash
make lint

# or
poetry run pre-commit run --all-files
```

## Troubleshooting

### Backend Verification

To verify that the backend is installed correctly, you can run the following command:

```bash
# Replace <backend> with the desired backend (e.g., cuda, rocm, mps, intel, cpu)
poetry run python scripts/backend_verification.py -eb <backend>
```

### Hardware Acceleration Issues

If you're experiencing issues with hardware acceleration:

1. **NVIDIA CUDA:**
   * Ensure you have the latest NVIDIA drivers installed
   * Check that `nvidia-smi` works in your terminal
   * Verify CUDA toolkit is properly installed

2. **AMD ROCm:**
   * Ensure ROCm is properly installed
   * Check that `rocminfo` works in your terminal

3. **Apple Silicon:**
   * Ensure you're using macOS 12.3 or later
   * Make sure you're using Python 3.8 or later

4. **Intel GPU:**
   * Ensure you have the latest Intel graphics drivers
   * Check that the Intel oneAPI Base Toolkit is installed if using SYCL

### Installation Errors

If the installation fails:

1. Check the output for specific error messages
2. Try running with verbose output: `make -n install` to see what commands would be executed
3. If Poetry installation is failing, try installing it manually first
4. For hardware detection issues, the `Makefile` handles this automatically.  If you *must* specify a backend, use the `Makefile`'s `BACKEND` variable:
    ```bash
    make install backend=<backend>  # Replace '<backend>' with your desired backend ("cuda", "rocm", "mps", "intel", or "cpu")
    ```

For further assistance, please open an issue on the project repository.
