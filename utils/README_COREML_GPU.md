# CoreML GPU Conversion for LLMs

This script converts HuggingFace models to CoreML format optimized for GPU (MPS) inference on Apple Silicon, **WITHOUT** the ANE-specific optimizations that ANEMLL uses.

## ⚠️ Important Limitations

**This script is experimental for large LLMs (30B+).** Direct CoreML conversion of full autoregressive language models is challenging because:

1. **Tracing complexity**: LLMs have dynamic computation graphs
2. **Memory requirements**: 30B models need 64GB+ RAM for conversion
3. **Limited CoreML LLM support**: CoreML is optimized for ANE, not primarily GPU

**For production GPU inference, we recommend:**
- ✅ **MLX** (recommended): `pip install mlx mlx-lm`
- ✅ **PyTorch MPS**: Use `model.to('mps')` directly
- ✅ **ONNX Runtime**: Export to ONNX first

## Installation

```bash
# Install dependencies
pip install -r utils/requirements_coreml_gpu.txt

# Or with uv (faster)
uv pip install -r utils/requirements_coreml_gpu.txt --index-url https://pypi.apple.com/simple
```

## Usage

### Basic Usage

```bash
# Convert a small model (for testing)
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-0.6B \
    --output ./models/qwen3-0.6b-gpu

# Convert with verbose logging (default)
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-30B-A3B \
    --output ./models/qwen3-30b-gpu
```

### Advanced Options

```bash
# Use FP32 instead of FP16 (larger, potentially more accurate)
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-30B-A3B \
    --output ./models/qwen3-30b-fp32 \
    --no-fp16

# Target CPU only (for testing)
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-0.6B \
    --output ./models/qwen3-cpu \
    --compute-unit CPU_ONLY

# Quiet mode (less verbose)
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-0.6B \
    --output ./models/qwen3 \
    --quiet

# Custom sequence length
python utils/convert_to_coreml_gpu.py \
    --model Qwen/Qwen3-0.6B \
    --output ./models/qwen3 \
    --max-seq-length 1024
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model name or path | Required |
| `--output` | Output directory for converted model | Required |
| `--compute-unit` | Target compute unit (CPU_AND_GPU, ALL, CPU_ONLY) | CPU_AND_GPU |
| `--no-fp16` | Use FP32 instead of FP16 | FP16 enabled |
| `--max-seq-length` | Maximum sequence length | 512 |
| `--quiet` | Less verbose output | Verbose |

## Expected Output

After successful conversion, you'll find:

```
output_directory/
├── model.mlpackage/          # CoreML model
├── tokenizer/                 # Tokenizer files
├── metadata.txt              # Conversion metadata
└── coreml_conversion.log     # Detailed logs
```

## System Requirements

### For Small Models (0.6B - 3B)
- **RAM**: 16GB minimum
- **Storage**: 20GB free space
- **Time**: 5-15 minutes

### For Large Models (7B - 30B)
- **RAM**: 64GB minimum (128GB recommended)
- **Storage**: 100GB+ free space
- **Time**: 30 minutes - 2 hours

## Using the Converted Model

```python
import coremltools as ct
from transformers import AutoTokenizer

# Load model
model = ct.models.MLModel('models/qwen3-30b-gpu/model.mlpackage')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/qwen3-30b-gpu/tokenizer')

# Prepare input
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='np')

# Run inference
output = model.predict({'input_ids': input_ids})
```

## Troubleshooting

### Conversion Fails: "Out of Memory"

**Problem**: Not enough RAM for model conversion

**Solution**:
1. Close other applications
2. Use smaller model for testing
3. Try FP16 (default, uses less memory)
4. Consider using MLX instead

### Conversion Fails: "Tracing Error"

**Problem**: LLM architecture too complex for tracing

**Solution**:
```bash
# Use MLX instead (recommended)
pip install mlx mlx-lm
python -m mlx_lm.generate --model Qwen/Qwen3-30B-A3B --prompt "Hello"
```

### Model Too Slow on GPU

**Problem**: CoreML model not optimized for your use case

**Solution**: Use PyTorch MPS directly:
```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    torch_dtype=torch.float16,
    device_map="mps"
)
```

## Why MLX is Better for LLM GPU Inference

MLX is specifically designed for LLM inference on Apple Silicon GPU:

```bash
# Install MLX
pip install mlx mlx-lm

# Run model on GPU (much simpler!)
python -m mlx_lm.generate \
    --model Qwen/Qwen3-30B-A3B \
    --max-tokens 500 \
    --prompt "Your prompt here"

# Or quantize for faster inference
python -m mlx_lm.generate \
    --model Qwen/Qwen3-30B-A3B \
    --max-tokens 500 \
    --prompt "Your prompt here" \
    --quantize  # Enables 4-bit quantization
```

**MLX Advantages:**
- ✅ Purpose-built for Apple Silicon GPU
- ✅ Native 4-bit/8-bit quantization
- ✅ Optimized for autoregressive generation
- ✅ Active development by Apple
- ✅ Easy to use (no conversion needed)
- ✅ Better performance than CoreML for LLMs

## Logs

All conversion logs are saved to `coreml_conversion.log` in the current directory.

```bash
# View logs in real-time
tail -f coreml_conversion.log

# Search for errors
grep "ERROR" coreml_conversion.log

# View full log
cat coreml_conversion.log
```

## Comparison: CoreML vs MLX vs PyTorch MPS

| Feature | CoreML GPU | MLX | PyTorch MPS |
|---------|-----------|-----|-------------|
| **Conversion** | Complex | None needed | None needed |
| **Speed** | Good | Excellent | Good |
| **Memory** | High during conversion | Efficient | Moderate |
| **Quantization** | Limited | Native 4/8-bit | Manual |
| **LLM Support** | Experimental | Excellent | Good |
| **Ease of Use** | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| **Recommendation** | Testing only | **Production** | Development |

## Real-World Recommendation

**For Qwen3-30B on Apple Silicon GPU, use MLX:**

```bash
# Install MLX (one-time)
pip install mlx mlx-lm

# Use immediately (no conversion!)
python -m mlx_lm.generate \
    --model Qwen/Qwen3-30B-A3B \
    --max-tokens 500 \
    --temp 0.7 \
    --prompt "Explain quantum computing in simple terms:"
```

This script is provided for educational purposes and experimentation. For production LLM inference on Apple Silicon GPU, **MLX is the recommended solution**.

## Support

For issues specific to:
- **This script**: Open issue on ANEMLL repo
- **CoreML**: Check [coremltools documentation](https://coremltools.readme.io/)
- **MLX**: Check [MLX documentation](https://ml-explore.github.io/mlx/)
- **HuggingFace models**: Check [Transformers documentation](https://huggingface.co/docs/transformers/)
