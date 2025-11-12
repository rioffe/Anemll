#!/bin/bash

# ANEMLL UV Environment Setup Script
# Creates a Python 3.11 virtual environment using uv package manager
# Requires: uv (https://github.com/astral-sh/uv)

set -e

echo "🚀 ANEMLL UV Environment Setup"
echo "==============================="

# Environment name
ENV_NAME="env-anemll-uv"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo ""
    echo "Please install uv first:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or with Homebrew: brew install uv"
    echo "  Or with pip: pip install uv"
    echo ""
    echo "For more information: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✅ Found uv: $(uv --version)"

# Check if the environment already exists
if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Found existing $ENV_NAME environment"
    read -p "Do you want to remove it and create a fresh environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        rm -rf "$ENV_NAME"
        echo "✅ Existing environment removed"
    else
        echo "❌ Aborting. Please remove the environment manually or choose a different name."
        exit 1
    fi
fi

# Create virtual environment with Python 3.11 using uv
echo ""
echo "📦 Creating virtual environment with Python 3.11..."
uv venv "$ENV_NAME" --python 3.11

# Verify creation
if [ ! -d "$ENV_NAME" ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created successfully!"

# Activate the environment
echo ""
echo "🔧 Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✅ Using $PYTHON_VERSION"

# Verify it's Python 3.11
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_MINOR" != "11" ]; then
    echo "⚠️  Warning: Expected Python 3.11, but got Python 3.$PYTHON_MINOR"
    echo "This may still work, but Python 3.11 is recommended for this setup."
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ UV environment setup complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Environment details:"
echo "  • Name: $ENV_NAME"
echo "  • Python: $PYTHON_VERSION"
echo "  • Package manager: uv"
echo ""
echo "Next steps:"
echo "  1. Install dependencies: ./install_dependencies_uv.sh"
echo "  2. Verify installation: python -c 'import torch; print(torch.__version__)'"
echo "  3. Test conversion: ./tests/conv/test_hf_model.sh Qwen/Qwen3-0.6B"
echo ""
echo "To activate this environment in future sessions:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "⚡ This environment uses uv for faster package installation!"
echo "════════════════════════════════════════════════════════"
