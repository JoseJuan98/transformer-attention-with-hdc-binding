# Setup Guide

This guide outlines the steps to set up the development environment for this project.

## Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.11 or higher:**  The project requires Python 3.11 or a later version. Download the appropriate installer from [python.org](https://www.python.org/downloads/).
*   **A Python Environment Manager:**  You'll need a tool to manage your Python environment and dependencies.  We recommend **Poetry** or **pip** with `virtualenv`, but other tools like `conda`, `uv`, `pipx` are also compatible.
*   **Make (Optional, but Recommended):**  The project includes a `Makefile` to automate common tasks like dependency installation and testing.  While optional, using `make` simplifies the setup process.

## Hardware Acceleration Support

This project provides comprehensive support for multiple hardware acceleration backends to optimize performance across different systems:

### Supported Platforms

*   **NVIDIA CUDA** - Full support for NVIDIA GPUs
    *   Default: CUDA 12.4
    *   Custom versions can be specified in the `[install.sh](../scripts/install.sh)` by modifying the `CUDA_VERSION` variable.

*   **AMD ROCm** - Optimized for AMD GPUs
    *   Default: ROCm 6.2.4
    *   Custom versions can be specified in the `[install.sh](../scripts/install.sh)` by modifying the `ROCM_VERSION` variable.

*   **Apple Silicon (MPS)** - Native acceleration for Apple M1/M2/M3 chips
    *   Leverages Metal Performance Shaders for optimal performance

*   **Intel GPU** - Support for Intel GPUs with SYCL
    *   Custom versions can be specified in the `[install.sh](../scripts/install.sh)` by modifying the `INTEL_EXTENSION_VERSION` variable.

*   **CPU-only** - Reliable fallback option
    *   Ensures compatibility on systems without supported GPUs or when troubleshooting hardware-specific issues

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

1.  **For standard installation:**

    ```bash
    make install
    ```

    This command will:
    *   Install Poetry if not already installed
    *   Detect your hardware (NVIDIA CUDA, AMD ROCm, Apple Silicon, Intel GPU, or CPU-only)
    *   Install the appropriate PyTorch backend for your hardware
    *   Set up the project environment

2.  **For development setup:**

    ```bash
    make install-dev
    ```

    This command does everything the standard installation does, plus:
    *   Installs development dependencies
    *   Sets up pre-commit hooks
    *   Installs testing tools

3.  **Run experiments:**

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

While the Makefile automates backend selection, you can still use Poetry directly, but it's less convenient. The Makefile is the preferred method.

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
4. For hardware detection issues, try specifying the backend manually:
   ```bash
   make poetry-install backend=cpu  # Replace 'cpu' with your desired backend
   ```

For further assistance, please open an issue on the project repository.
