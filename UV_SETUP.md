# UV Setup - Quick Reference (Apple Internal)

Fast alternative setup using uv package manager with Python 3.11, PyTorch 2.7.0, and Apple's internal PyPI.

## Quick Start (3 commands)

```bash
./create_uv_env.sh                    # Create Python 3.11 environment
./install_dependencies_uv.sh           # Install dependencies from Apple PyPI (fast!)
python tests/test_qwen_model.py       # Verify installation
```

## What You Get

- ⚡ **10-100x faster** package installation
- 🐍 **Python 3.11** - Modern Python with better performance
- 🔥 **PyTorch 2.7.0** - Latest stable release
- 📦 **uv** - Fast package manager from Astral
- 🍎 **Apple PyPI** - Internal package repository (https://pypi.apple.com/simple)

## Prerequisites

Install uv first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Files

- `create_uv_env.sh` - Creates Python 3.11 virtual environment using uv
- `install_dependencies_uv.sh` - Installs all dependencies from Apple PyPI with uv
- `requirements-uv.txt` - Pinned dependencies for reproducibility
- `docs/uv_setup.md` - Complete documentation and troubleshooting

## Manual Package Installation

If you need to install additional packages:

```bash
# Activate environment first
source env-anemll-uv/bin/activate

# Install from Apple PyPI
uv pip install <package-name> --index-url https://pypi.apple.com/simple
```

## Daily Usage

```bash
# Activate environment
source env-anemll-uv/bin/activate

# Convert a model (same as standard setup)
./anemll/utils/convert_model.sh --model <model> --output <dir>

# Run tests (same as standard setup)
python tests/chat.py --meta <model>/meta.yaml

# Deactivate when done
deactivate
```

## Comparison with Standard Setup

| Aspect | Standard | UV Setup |
|--------|----------|----------|
| Python | 3.9 | 3.11 |
| PyTorch | 2.5.0 | 2.7.0 |
| Manager | pip | uv |
| PyPI | Public | Apple Internal |
| Speed | ~4 min | ~40 sec |
| Environment | `env-anemll` | `env-anemll-uv` |

## Documentation

Full guide: [`docs/uv_setup.md`](docs/uv_setup.md)

## Important Notes

- **Apple PyPI**: All packages are installed from `https://pypi.apple.com/simple`
- **Package Availability**: Some packages (like `lm-evaluation-harness`) may not be in Apple PyPI
  - These will be skipped automatically during installation
  - You can install them manually from public PyPI if needed:
    ```bash
    uv pip install lm-evaluation-harness --index-url https://pypi.org/simple
    ```
- **Compatibility**: All ANEMLL commands work identically between setups
- **Models**: Converted models are compatible between standard and UV setups
- **Coexistence**: This setup can coexist with the standard setup (different environment names)

---

**Note**: This is an alternative setup for Apple internal use. The standard Python 3.9 + pip setup remains fully supported.
