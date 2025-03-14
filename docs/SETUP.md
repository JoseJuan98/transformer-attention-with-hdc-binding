# Setup Guide

This guide outlines the steps to set up the development environment for this project.

## Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.11 or higher:**  The project requires Python 3.11 or a later version. Download the appropriate installer from [python.org](https://www.python.org/downloads/).
*   **A Python Environment Manager:**  You'll need a tool to manage your Python environment and dependencies.  We recommend **Poetry** or **pip** with `virtualenv`, but other tools like `conda`, `uv`, `pipx` are also compatible.
*   **Make (Optional, but Recommended):**  The project includes a `Makefile` to automate common tasks like dependency installation and testing.  While optional, using `make` simplifies the setup process.

## Installing Make (Optional)

`make` is a build automation tool.  If you choose to use it, follow the instructions below for your operating system:

*   **Windows:**
    1.  Open PowerShell as an administrator.
    2.  Run: `winget install GnuWin32.Make`
    3.  Add the installation directory (typically `C:\Program Files (x86)\GnuWin32\bin`) to your system's `PATH` environment variable.  You can do this by searching for "environment variables" in the Windows search bar, clicking "Edit the system environment variables," then "Environment Variables...", selecting "Path" under "System variables," clicking "Edit," and adding the path.
*   **macOS:**
    1.  Open a terminal.
    2.  Run: `brew install make` (This requires [Homebrew](https://brew.sh/) to be installed).
*   **Linux:** `make` is usually pre-installed on most Linux distributions.  You can verify by running `make --version` in your terminal. If it's not installed, use your distribution's package manager (e.g., `apt-get install make` on Debian/Ubuntu, `yum install make` on Fedora/CentOS).

## Setting Up the Python Environment

Choose *one* of the following methods to set up your Python environment:

### Method 1: Using Poetry (Recommended)

Poetry simplifies dependency management and packaging.

1.  **Install Poetry (if you don't have it):**

    ```bash
    pip install --no-cache-dir -U "poetry>=2.1.1"
    ```

2.  **Install Dependencies and Activate the Environment:**

    *   **Using `make` (Recommended):**

        ```bash
        make install  # Installs project dependencies
        ```
        or
        ```bash
        make install-dev # Installs project and dev dependencies, and pre-commit hooks
        ```

    *   **Without `make`:**

        ```bash
        poetry install --with=dev  # Installs project and dev dependencies
        ```

### Method 2: Using pip and virtualenv

1.  **Install `virtualenv`:**

    ```bash
    pip install --no-cache-dir virtualenv
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python3 -m virtualenv .venv --python="C:\Program Files\python3.12\python.exe"
    ```
    *   Usually the Python paths are `C:\Program Files\python3.12\python.exe` on Windows, `/usr/bin/python3.12` on Linux/macOS).  The exact path may vary depending on your installation.  You can find the path by running `where python3.12` (Windows) or `which python3.12` (Linux/macOS).

3.  **Activate the Virtual Environment:**

    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**

    ```bash
    pip install --no-cache-dir -e .[dev]
    ```

## Setting Up Automatic Formatting

To automatically format your code according to the project's style (Black), follow these steps:

1.  **Install Development Dependencies:** Ensure you've installed the development dependencies (using either `make install-dev`, `poetry install --with=dev`, or `pip install --no-cache-dir -e .[dev]`). This will install the `black` formatter.

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

## Pre-commit Hooks (Optional, but Recommended)

If you installed the development dependencies, the pre-commit hooks should be installed. It's recommended to run these hooks before committing changes to ensure code quality.

To check that they are installed correctly, run:

```bash
make lint

# or
pre-commit run --all-files
```
