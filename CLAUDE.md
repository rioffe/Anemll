# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANEMLL (pronounced "animal") is an open-source project for accelerating Large Language Models (LLMs) on Apple Neural Engine (ANE). The project converts Hugging Face models to CoreML format for on-device inference on Apple devices.

## Development Commands

### Environment Setup
```bash
# Automated setup (recommended)
./create_python39_env.sh          # Creates Python 3.9 virtual environment
./install_dependencies.sh          # Installs dependencies (auto-detects venv)

# Manual setup
python3.9 -m venv env-anemll
source env-anemll/bin/activate
pip install -r requirements.txt

# Verify installation
xcode-select --install             # Required for CoreML compilation
xcrun --find coremlcompiler        # Verify CoreML compiler
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Important**: Always activate the virtual environment before running any Python scripts:
```bash
source env-anemll/bin/activate  # or anemll-env/bin/activate
```

Verify the environment is active:
- Prompt shows `(env-anemll)` or `(anemll-env)`
- `which python` points to virtual environment
- `python --version` shows Python 3.9.x

### Model Conversion

```bash
# Single-shot conversion script (main entry point)
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory>

# With quantization and chunking
./anemll/utils/convert_model.sh \
    --model ./models/llama-3.1-1b \
    --output ./converted_models \
    --context 512 \
    --batch 64 \
    --lut2 4 \      # FFN quantization (4-bit)
    --lut3 6 \      # LM head quantization (6-bit)
    --chunk 2       # Number of FFN chunks
```

The conversion script automatically:
1. Detects model architecture (LLaMA, Qwen, DeepSeek)
2. Converts embeddings, FFN, and LM head separately
3. Applies LUT quantization (4-bit for FFN, 6-bit for LM head)
4. Chunks large models to fit ANE constraints
5. Compiles to CoreML format
6. Creates meta.yaml configuration

### Testing

```bash
# Quick model tests (downloads models automatically)
python tests/test_qwen_model.py      # Test Qwen 3 conversion
python tests/test_qwen2.5_model.py   # Test Qwen 2.5 conversion
python tests/test_llama_model.py     # Test LLaMA conversion

# Generic HuggingFace model testing
./tests/conv/test_hf_model.sh <model_name> [output_dir] [chunks]

# Examples:
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-1B-Instruct
./tests/conv/test_hf_model.sh Qwen/Qwen2.5-0.5B-Instruct /tmp/my-test
./tests/conv/test_hf_model.sh meta-llama/Llama-3.2-8B-Instruct /tmp/llama8b 4
```

### Chat Interfaces

```bash
# Basic chat (quick testing)
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full conversation mode (maintains history, handles context window)
python ./tests/chat_full.py --meta ./converted_models/meta.yaml

# Manual model specification
python ./tests/chat.py \
    --embed llama_embeddings \
    --lmhead llama_lm_head_lut6 \
    --ffn llama_FFN_PF_lut4_chunk_01of02 \
    --tokenizer ./converted_models \
    --context-length 512 \
    --d ./converted_models
```

### Model Evaluation

```bash
# Evaluate with lm-evaluation-harness
python evaluate/ane/evaluate_with_harness.py \
    --model /path/to/model \
    --tasks boolq,arc_easy,hellaswag \
    --batch-size 1 \
    --num-shots 0

# Run evaluation suite
./run_ane_evaluations.sh <model_path> <output_dir>
```

### Swift CLI Development

```bash
# Build Swift CLI
cd anemll-swift-cli
swift build

# Run Swift CLI
swift run anemllcli --help

# Run tests
swift test
```

## Architecture Overview

### Core Components

1. **ANE Converter Pipeline** (`anemll/ane_converter/`)
   - `base_converter.py`: Abstract base class defining conversion interface
   - `llama_converter.py`: LLaMA/DeepSeek model conversion
   - `qwen_converter.py`: Qwen 3 model conversion
   - `qwen2_5_converter.py`: Qwen 2.5 model conversion
   - `deepseek_converter.py`: DeepSeek-specific optimizations
   - `metadata.py`: Model metadata and configuration
   - `optimization_rules.py`: ANE-specific optimization rules

2. **Model Implementations** (`anemll/models/`)
   - `base_model.py`: Abstract base with weight loading interface
   - `llama_model.py`: LLaMA architecture (reference implementation)
   - `qwen_model.py`: Qwen 3 architecture
   - `qwen2_5_model.py`: Qwen 2.5 architecture
   - `deepseek_model.py`: DeepSeek architecture

3. **Utilities** (`anemll/utils/`)
   - `convert_model.sh`: Main conversion orchestration script
   - `combine_models.py`: Combines chunked FFN models
   - `compile_models.py`: CoreML compilation with LUT quantization
   - `generate_meta_yaml.py`: Creates deployment configuration

4. **Swift Implementation** (`anemll-swift-cli/`)
   - `InferenceManager.swift`: Manages model inference pipeline
   - `ModelLoader.swift`: Loads and manages CoreML models
   - `Tokenizer.swift`: Tokenization handling
   - `YAMLConfig.swift`: Configuration file parsing
   - `FFNChunk.swift`: FFN chunk management

5. **iOS/macOS Sample App** (`anemll-chatbot/`)
   - SwiftUI-based chat interface
   - Model downloading and management
   - CoreML inference integration

6. **Evaluation Infrastructure** (`evaluate/ane/`)
   - `evaluate_with_harness.py`: lm-evaluation-harness integration
   - Direct CoreML integration with ANE models
   - Support for standard benchmarks (BoolQ, ARC, HellaSwag, etc.)

### Conversion Pipeline (8-Step Process)

The model conversion follows this workflow:

1. **Convert Embeddings** (part 1) - Token embeddings with optional LUT quantization
2. **Convert LM Head** (part 3) - Language model head with LUT quantization (typically 6-bit)
3. **Convert FFN Layers** (part 2) - Feed-forward network with chunking and LUT quantization (typically 4-bit)
4. **Convert Prefill Attention** (part 2_prefill) - Attention mechanism for prefill phase
5. **Combine Chunked Models** - Merges FFN chunks into unified model
6. **Compile to CoreML** - Generates .mlmodelc or .mlpackage files
7. **Create Configuration** - Copies tokenizer files and creates meta.yaml
8. **Test with Chat** - Validates with chat interface

### Key Design Patterns

- **Multi-part Architecture**: Models split into 3 parts for ANE optimization (embeddings, FFN/attention, LM head)
- **Chunking Strategy**: FFN layers chunked to fit ANE memory constraints (typical: 2-4 chunks)
- **LUT Quantization**: Lookup table quantization (4-bit for FFN, 6-bit for LM head, optional for embeddings)
- **Meta Configuration**: YAML-based deployment configuration for easy model loading
- **State Management**: Separate KV cache handling for efficient sequential generation

### ANE-Specific Implementation Requirements

**CRITICAL**: When implementing models for ANE (Apple Neural Engine) compatibility:

#### 1. RMSNorm Implementation (REQUIRED)

**NEVER** implement "true" RMSNorm (variance-only normalization) - it causes precision issues on ANE.

**ALWAYS** follow this pattern from `llama_model.py`:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    mean = hidden_states.mean(-1, keepdim=True)
    hidden_states = hidden_states - mean
    return F.layer_norm(hidden_states, self.weight.shape, self.weight, bias=None, eps=float(self.eps)).to(TEST_DEVICE).to(MODEL_DTYPE)
```

Why this matters:
- Subtracts mean FIRST (converts to LayerNorm-like operation)
- Uses `F.layer_norm()` instead of manual variance computation
- Standard RMSNorm without mean subtraction will fail on ANE

#### 2. Conv2d Layer Requirements

All dense layers MUST be `nn.Conv2d` with `kernel_size=(1,1)` - NEVER use `nn.Linear`:

```python
# Correct for ANE
self.q_proj = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 1))

# Incorrect - will not run on ANE
self.q_proj = nn.Linear(hidden_size, hidden_size)
```

#### 3. Tensor Shape Constraints

- Tensor ranks MUST be ≤4 with dimensions `(N, C, H, W)`
- Height/Width dimensions MUST NOT exceed 16,384 elements
- Channel dimension MUST NOT exceed 65,536 elements
- Maintain trailing dimension ≥64 for better ANE tiling

#### 4. Weight Reshaping

Reshape HuggingFace weights from `(out, in)` to `(out, in, 1, 1)` for Conv2d:

```python
# Loading weights from HuggingFace
hf_weight = hf_model.weight  # Shape: (out_features, in_features)
conv_weight = hf_weight.unsqueeze(-1).unsqueeze(-1)  # Shape: (out, in, 1, 1)
```

#### 5. Device and Dtype Management

Always use `.to(TEST_DEVICE).to(MODEL_DTYPE)` at the end of forward passes:

```python
# Correct
output = F.layer_norm(hidden_states, ...).to(TEST_DEVICE).to(MODEL_DTYPE)

# Required constants
MODEL_DTYPE = torch.float16  # Must be float16 throughout
```

- Initialize all parameters on correct device in `__init__`
- Maintain consistent dtype (float16) across entire pipeline
- Never let tensors drift to different devices or dtypes

### Testing Infrastructure

The project includes extensive testing organized in `tests/dev/`:

- **KV Cache Testing**: `test_kv_cache_*.py`, `debug_kv_*.py`
- **CoreML vs PyTorch Comparison**: `test_pytorch_vs_coreml.py`, `test_coreml_vs_pytorch.py`
- **Sequential Generation**: `test_sequential_tokens.py`, `test_coreml_sequential.py`
- **Architecture-Specific**: `test_llama_model.py`, `test_qwen_model.py`
- **Attention Mechanisms**: `debug_attention_*.py`, `debug_rotary_emb.py`

See `tests/dev/README.md` for complete catalog.

## Development Guidelines

### Test and Debug File Organization

**IMPORTANT**: Always create test, debug, and development files in `./tests/dev/` to keep the root directory clean.

When working on:
- **Bug fixes**: Create debug scripts in `./tests/dev/debug_<issue_name>.py`
- **New architecture support**: Create test files in `./tests/dev/test_<arch>_<feature>.py`
- **Model validation**: Create comparison scripts in `./tests/dev/test_<model>_vs_<reference>.py`
- **Development utilities**: Place tools in `./tests/dev/` with descriptive names

**Never** create test or debug files directly in the root directory.

### Adding New Model Architecture Support

1. **Copy Closest Existing Model**
   - Start with `llama_model.py` for transformer-based architectures
   - Rename all classes to match new architecture

2. **Implement ANE-Compatible Layers**
   - Use Conv2d instead of Linear
   - Follow RMSNorm mean-subtraction pattern
   - Ensure device/dtype preservation throughout

3. **Create Converter**
   - Subclass `BaseConverter` in `anemll/ane_converter/`
   - Implement weight loading and reshaping
   - Handle architecture-specific optimizations

4. **Test Numerical Parity**
   - Compare against HuggingFace reference implementation
   - Verify KV cache correctness
   - Test sequential generation

5. **Update Documentation**
   - Add to supported models list
   - Document any architecture-specific quirks

### Code Quality Requirements

- Format with `black` and `ruff` - zero linter warnings
- Write unit tests verifying shape parity and deterministic inference
- Test without internet access requirement (use cached models)
- Follow existing patterns from `llama_model.py`

### Common Pitfalls to Avoid

- **Don't modify `llama_model.py`** - it's the reference implementation
- **Don't subclass new models from LLaMA** - copy and rename instead
- **Don't use "true" RMSNorm** - always subtract mean first
- **Don't use Linear layers** - use Conv2d with kernel_size=(1,1)
- **Don't mix dtypes** - maintain float16 throughout
- **Don't create files in root** - use `tests/dev/` for development

## System Requirements

- **System**: macOS Sequoia with Apple Neural Engine (Apple Silicon)
- **Memory**: Minimum 16GB RAM (32GB recommended for 8B models)
- **Python**: 3.9-3.11 (3.9 strongly recommended for best compatibility)
- **Tools**: Xcode Command Line Tools for coremlcompiler
- **Dependencies**: coremltools>=8.2, transformers>=4.36.0, numpy>=1.24.0, scikit-learn<=1.5.1

## Model Support

### Fully Supported (Stable)
- **LLaMA 3.1/3.2** (1B, 8B variants)
- **DeepSeek R1** (8B distilled)
- **DeepHermes** (3B, 8B)

### Experimental (Alpha)
- **Qwen 3** (0.6B, 1.7B, 4B)
- **Qwen 2.5** (0.5B-Instruct, 1.5B, 3B, 7B)

### Context Lengths
- **Recommended**: 512-1024 tokens for optimal ANE performance
- **Verified**: Up to 4K tokens
- **Theoretical**: Qwen supports up to 32K

### Pre-converted Models
Available at https://huggingface.co/anemll:
- iOS-friendly builds (unzipped .mlmodelc)
- Standard builds for macOS development
- Multiple quantization levels (FP16, LUT4, LUT6)
