#!/bin/bash

# ANEMLL UV Dependencies Installation Script
# Installs all required dependencies using uv package manager
# Includes PyTorch 2.7.0 and Python 3.11 optimizations
# Uses Apple internal PyPI: https://pypi.apple.com/simple

set -e

echo "🚀 Installing ANEMLL Dependencies with UV..."
echo "=============================================="
echo "📍 Using Apple PyPI: https://pypi.apple.com/simple"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in the correct virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    # Try to activate env-anemll-uv if it exists
    if [[ -f "./env-anemll-uv/bin/activate" ]]; then
        echo "🔄 Activating env-anemll-uv virtual environment..."
        source ./env-anemll-uv/bin/activate
    else
        echo "❌ Error: No virtual environment detected"
        echo "Please create and activate a virtual environment first:"
        echo "  ./create_uv_env.sh"
        exit 1
    fi
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"

# Verify Python version
PYTHON_VERSION=$(python --version)
PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')

echo "📍 Detected Python version: $PYTHON_VERSION"

if [[ $PYTHON_MAJOR -ne 3 || $PYTHON_MINOR -lt 9 ]]; then
    echo "❌ ERROR: ANEMLL requires Python 3.9 or higher"
    echo "Current Python version is $PYTHON_VERSION"
    exit 1
fi

if [[ $PYTHON_MINOR -eq 11 ]]; then
    echo "✅ Using Python 3.11 (optimized for uv setup)"
elif [[ $PYTHON_MINOR -ge 12 ]]; then
    echo "⚠️  WARNING: Python 3.$PYTHON_MINOR detected"
    echo "This setup is optimized for Python 3.11"
    echo "Python 3.12+ may require additional compatibility adjustments"
else
    echo "ℹ️  Using Python 3.$PYTHON_MINOR (setup optimized for 3.11)"
fi

echo ""
echo "📦 Installing dependencies with uv (fast!)"
echo "=========================================="

# Set Apple PyPI index
APPLE_PYPI="https://pypi.apple.com/simple"

# Upgrade pip first (uv uses pip internally for some operations)
echo "📦 Upgrading pip..."
uv pip install --upgrade pip --index-url $APPLE_PYPI

# Install PyTorch 2.7.0
echo ""
echo "🔥 Installing PyTorch 2.7.0..."
echo "   (This may take a few minutes on first install)"
uv pip install torch==2.7.0 torchvision torchaudio --index-url $APPLE_PYPI

# Install CoreML Tools (must be after PyTorch)
echo ""
echo "🧠 Installing CoreML Tools..."
uv pip install "coremltools>=8.2" --index-url $APPLE_PYPI

# Install core ANEMLL dependencies
echo ""
echo "📚 Installing core dependencies..."
uv pip install "transformers>=4.39.0" --index-url $APPLE_PYPI
uv pip install "numpy>=1.24.0" --index-url $APPLE_PYPI
uv pip install "scikit-learn<=1.5.1" --index-url $APPLE_PYPI  # Required for LUT quantization
uv pip install datasets --index-url $APPLE_PYPI
uv pip install accelerate --index-url $APPLE_PYPI
uv pip install safetensors --index-url $APPLE_PYPI
uv pip install tokenizers --index-url $APPLE_PYPI
uv pip install sentencepiece --index-url $APPLE_PYPI
uv pip install pyyaml --index-url $APPLE_PYPI

# Install evaluation dependencies
echo ""
echo "📊 Installing evaluation dependencies..."
uv pip install "lm-evaluation-harness>=0.4.9" --index-url $APPLE_PYPI

# Install development dependencies
echo ""
echo "🛠️  Installing development dependencies..."
uv pip install black --index-url $APPLE_PYPI
uv pip install flake8 --index-url $APPLE_PYPI
uv pip install pytest --index-url $APPLE_PYPI
uv pip install pytest-cov --index-url $APPLE_PYPI
uv pip install jupyter --index-url $APPLE_PYPI
uv pip install ipykernel --index-url $APPLE_PYPI

# Install optional but recommended dependencies
echo ""
echo "⚡ Installing optional dependencies..."
uv pip install huggingface_hub --index-url $APPLE_PYPI
uv pip install tqdm --index-url $APPLE_PYPI
uv pip install matplotlib --index-url $APPLE_PYPI
uv pip install seaborn --index-url $APPLE_PYPI
uv pip install psutil --index-url $APPLE_PYPI

# Install ANEMLL package in development mode
if [ -f "pyproject.toml" ]; then
    echo ""
    echo "📦 Installing ANEMLL package in development mode..."
    uv pip install -e . --index-url $APPLE_PYPI
elif [ -f "setup.py" ]; then
    echo ""
    echo "📦 Installing ANEMLL package in development mode..."
    uv pip install -e . --index-url $APPLE_PYPI
fi

# Verify installations
echo ""
echo "════════════════════════════════════════════════════════"
echo "🔍 Verifying installations..."
echo "════════════════════════════════════════════════════════"
echo ""

# Verify Python
echo -n "Python: "
python -c "import sys; print(f'✅ {sys.version.split()[0]}')" 2>/dev/null || echo "❌ Failed"

# Verify PyTorch
echo -n "PyTorch: "
python -c "import torch; print(f'✅ {torch.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify PyTorch MPS support (for Apple Silicon)
echo -n "PyTorch MPS: "
python -c "import torch; print('✅ Available' if torch.backends.mps.is_available() else '⚠️  Not available')" 2>/dev/null || echo "❌ Failed"

# Verify CoreML Tools
echo -n "CoreML Tools: "
python -c "import coremltools; print(f'✅ {coremltools.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify Transformers
echo -n "Transformers: "
python -c "import transformers; print(f'✅ {transformers.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify NumPy
echo -n "NumPy: "
python -c "import numpy; print(f'✅ {numpy.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify scikit-learn
echo -n "scikit-learn: "
python -c "import sklearn; print(f'✅ {sklearn.__version__}')" 2>/dev/null || echo "❌ Failed"

# Verify lm-evaluation-harness
echo -n "lm-eval-harness: "
python -c "import lm_eval; print(f'✅ {lm_eval.__version__}')" 2>/dev/null || echo "❌ Failed"

# Check Xcode Command Line Tools
echo ""
echo "🔧 Checking Xcode Command Line Tools..."
if xcode-select -p &> /dev/null; then
    echo "✅ Xcode Command Line Tools installed"
    if xcrun --find coremlcompiler &> /dev/null; then
        echo "✅ CoreML compiler found"
    else
        echo "⚠️  CoreML compiler not found"
        echo "   You may need to install Xcode Command Line Tools:"
        echo "   xcode-select --install"
    fi
else
    echo "⚠️  Xcode Command Line Tools not found"
    echo "   Required for CoreML compilation. Install with:"
    echo "   xcode-select --install"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ Installation complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Python: $PYTHON_VERSION"
echo "  • PyTorch: 2.7.0"
echo "  • Package manager: uv"
echo "  • PyPI index: $APPLE_PYPI"
echo "  • Environment: $VIRTUAL_ENV"
echo ""
echo "Next steps:"
echo "  1. Test conversion: ./tests/conv/test_hf_model.sh Qwen/Qwen3-0.6B"
echo "  2. Run chat interface: python tests/chat.py --meta <model_dir>/meta.yaml"
echo "  3. Convert a model: ./anemll/utils/convert_model.sh --help"
echo ""
echo "For more information, see: docs/uv_setup.md"
echo "════════════════════════════════════════════════════════"
