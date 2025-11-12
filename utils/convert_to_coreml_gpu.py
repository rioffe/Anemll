#!/usr/bin/env python3
"""
CoreML GPU Conversion Script for Large Language Models

Converts HuggingFace models to CoreML format optimized for GPU (MPS) inference
on Apple Silicon. Unlike ANEMLL's ANE-optimized conversion, this preserves
standard model architecture for better GPU performance.

Usage:
    python convert_to_coreml_gpu.py --model Qwen/Qwen3-30B-A3B --output ./models/qwen3-30b-gpu

Requirements:
    - coremltools>=8.2
    - transformers>=4.39.0
    - torch>=2.0.0
    - loguru
"""

import argparse
import os
import sys
import gc
import psutil
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

# Try to import coremltools early to give better error messages
try:
    import coremltools as ct
except ImportError as e:
    print(f"ERROR: coremltools not installed. Run: pip install coremltools>=8.2")
    print(f"Details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to import coremltools: {e}")
    print(f"You have coremltools installed but there's an import error.")
    print(f"Try: pip install --upgrade coremltools")
    sys.exit(1)

# Import loguru after coremltools check
from loguru import logger

# Configure basic logging immediately (will be reconfigured later in main)
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<level>{level}</level>: {message}", level="DEBUG")

# Check coremltools version
try:
    ct_version = tuple(map(int, ct.__version__.split('.')[:2]))
    logger.debug(f"Detected coremltools version: {ct.__version__}")

    if ct_version >= (9, 0):
        logger.warning(
            f"You have coremltools {ct.__version__}. "
            "This script was tested with 8.x. Some APIs may have changed."
        )
except Exception as e:
    logger.warning(f"Could not parse coremltools version: {e}")

# Now import CoreML components with version-aware handling
try:
    # Try coremltools 8.x import first
    from coremltools.models import ComputeUnit
except ImportError:
    try:
        # Try coremltools 9.x import location
        from coremltools import ComputeUnit
    except ImportError as e:
        print(f"ERROR: Failed to import ComputeUnit from coremltools: {e}")
        print(f"Detected coremltools version: {ct.__version__}")
        print(f"This script was tested with coremltools 8.x")
        print(f"For coremltools 9.x, imports may have changed.")
        print(f"Try downgrading: pip install 'coremltools>=8.2,<9.0'")
        sys.exit(1)
except AttributeError as e:
    print(f"ERROR: ComputeUnit not found in coremltools: {e}")
    print(f"Detected coremltools version: {ct.__version__}")
    print(f"This script was tested with coremltools 8.x")
    print(f"You may need to adjust imports for coremltools 9.0")
    sys.exit(1)


class CoreMLGPUConverter:
    """Converts HuggingFace models to CoreML format optimized for GPU."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        compute_unit: str = "CPU_AND_GPU",
        fp16: bool = True,
        max_seq_length: int = 512,
    ):
        """
        Initialize the converter.

        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save converted model
            compute_unit: Target compute unit (CPU_AND_GPU, ALL, CPU_ONLY)
            fp16: Use FP16 precision (smaller, faster on GPU)
            max_seq_length: Maximum sequence length for conversion
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.compute_unit = self._parse_compute_unit(compute_unit)
        self.fp16 = fp16
        self.max_seq_length = max_seq_length

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")

    def _parse_compute_unit(self, compute_unit: str) -> ComputeUnit:
        """Parse compute unit string to ComputeUnit enum."""
        compute_units = {
            "CPU_AND_GPU": ComputeUnit.CPU_AND_GPU,
            "ALL": ComputeUnit.ALL,
            "CPU_ONLY": ComputeUnit.CPU_ONLY,
        }

        cu = compute_units.get(compute_unit.upper())
        if cu is None:
            logger.warning(
                f"Unknown compute unit '{compute_unit}', defaulting to CPU_AND_GPU"
            )
            return ComputeUnit.CPU_AND_GPU

        logger.info(f"Target compute unit: {compute_unit}")
        return cu

    def check_system_resources(self) -> bool:
        """Check if system has sufficient resources."""
        logger.info("Checking system resources...")

        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)

        logger.info(f"System memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")

        if available_gb < 16:
            logger.warning(
                f"Low available memory ({available_gb:.1f}GB). "
                "Conversion may fail for large models. Recommend 32GB+ for 30B models."
            )

        # Check if MPS is available
        if torch.backends.mps.is_available():
            logger.success("Metal Performance Shaders (MPS/GPU) available ✓")
        else:
            logger.warning(
                "MPS not available. Model will target GPU but cannot verify on this system."
            )

        # Check disk space
        disk = psutil.disk_usage(str(self.output_dir))
        free_gb = disk.free / (1024**3)
        logger.info(f"Disk space: {free_gb:.1f}GB free")

        if free_gb < 50:
            logger.warning(
                f"Low disk space ({free_gb:.1f}GB). Large models may need 100GB+ for conversion."
            )
            return False

        return True

    def load_model_and_tokenizer(self) -> Tuple[torch.nn.Module, AutoTokenizer, AutoConfig]:
        """Load HuggingFace model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        logger.info("This may take several minutes for large models...")

        try:
            # Load config first to check model size
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info(f"Model config loaded: {config.model_type}")

            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters / 1e9
                logger.info(f"Model size: ~{num_params:.1f}B parameters")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.success(f"Tokenizer loaded: {len(tokenizer)} tokens in vocabulary")

            # Determine dtype
            dtype = torch.float16 if self.fp16 else torch.float32
            logger.info(f"Using dtype: {dtype}")

            # Load model
            logger.info("Loading model weights (this will take time for large models)...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Important for large models
            )

            # Put model in eval mode
            model.eval()
            logger.success("Model loaded successfully")

            # Log model architecture summary
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

            return model, tokenizer, config

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def trace_model(self, model: torch.nn.Module) -> torch.jit.ScriptModule:
        """
        Trace the model for conversion.

        Note: Full model tracing is complex for LLMs. This is a simplified approach.
        For production use, consider converting specific components or using MLX.
        """
        logger.info("Preparing model for tracing...")

        # Create example inputs
        batch_size = 1
        seq_length = min(self.max_seq_length, 128)  # Use smaller for tracing

        logger.info(f"Creating example inputs: batch_size={batch_size}, seq_length={seq_length}")

        example_input_ids = torch.randint(
            0, 1000, (batch_size, seq_length), dtype=torch.long
        )
        example_attention_mask = torch.ones(
            (batch_size, seq_length), dtype=torch.long
        )

        logger.info("Tracing model (this may take a while)...")
        logger.warning(
            "Note: Full LLM tracing is experimental. "
            "Consider using MLX or PyTorch MPS for production inference."
        )

        try:
            with torch.no_grad():
                # Attempt to trace the model
                # This is simplified and may not work for all model architectures
                traced_model = torch.jit.trace(
                    model,
                    (example_input_ids, example_attention_mask),
                    strict=False
                )
                logger.success("Model tracing completed")
                return traced_model

        except Exception as e:
            logger.error(f"Model tracing failed: {e}")
            logger.info(
                "Tip: Large LLMs are difficult to convert to CoreML. "
                "Consider using MLX (pip install mlx mlx-lm) for GPU inference instead."
            )
            raise

    def convert_to_coreml(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer
    ) -> ct.models.MLModel:
        """Convert PyTorch model to CoreML format."""
        logger.info("Starting CoreML conversion...")
        logger.info("This process may take 30+ minutes for large models")

        try:
            # For LLMs, we need a different approach since full model tracing is complex
            logger.warning(
                "Direct CoreML conversion of full LLM is experimental. "
                "This script demonstrates the approach but may not succeed for 30B models."
            )

            # Create example inputs for conversion
            batch_size = 1
            seq_length = min(self.max_seq_length, 128)

            example_input = torch.randint(0, 1000, (batch_size, seq_length))

            logger.info("Converting model to TorchScript...")
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input, strict=False)

            logger.info("Converting TorchScript to CoreML...")
            logger.info(f"Target compute unit: {self.compute_unit}")

            # Convert to CoreML
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input_ids", shape=(batch_size, seq_length))],
                convert_to="mlprogram",  # Use ML Program (supports more ops than neuralnetwork)
                compute_units=self.compute_unit,
                minimum_deployment_target=ct.target.macOS13,
            )

            logger.success("CoreML conversion completed!")
            return mlmodel

        except Exception as e:
            logger.error(f"CoreML conversion failed: {e}")
            logger.info(
                "\nAlternative approaches:\n"
                "1. Use MLX: pip install mlx mlx-lm (recommended for Apple Silicon GPU)\n"
                "2. Use PyTorch MPS: model.to('mps') for direct GPU inference\n"
                "3. Convert model components separately instead of full model\n"
                "4. Use ONNX as intermediate format: model -> ONNX -> CoreML"
            )
            raise

    def save_model(self, mlmodel: ct.models.MLModel, tokenizer: AutoTokenizer):
        """Save CoreML model and tokenizer."""
        logger.info("Saving converted model...")

        # Save CoreML model
        model_path = self.output_dir / "model.mlpackage"
        mlmodel.save(str(model_path))
        logger.success(f"CoreML model saved: {model_path}")

        # Save tokenizer
        tokenizer_path = self.output_dir / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        logger.success(f"Tokenizer saved: {tokenizer_path}")

        # Save metadata
        metadata_path = self.output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Compute Unit: {self.compute_unit}\n")
            f.write(f"FP16: {self.fp16}\n")
            f.write(f"Max Sequence Length: {self.max_seq_length}\n")
        logger.success(f"Metadata saved: {metadata_path}")

        # Print model info
        logger.info("Model information:")
        logger.info(f"  Size: {self._get_directory_size(model_path):.2f} MB")
        logger.info(f"  Location: {model_path.absolute()}")

    def _get_directory_size(self, path: Path) -> float:
        """Get directory size in MB."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            return 0
        return total / (1024 * 1024)

    def convert(self):
        """Main conversion workflow."""
        logger.info("=" * 70)
        logger.info("CoreML GPU Conversion Script")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output: {self.output_dir.absolute()}")
        logger.info("=" * 70)

        try:
            # Step 1: Check system resources
            logger.info("\n[Step 1/5] Checking system resources")
            if not self.check_system_resources():
                logger.warning("Proceeding despite resource warnings...")

            # Step 2: Load model and tokenizer
            logger.info("\n[Step 2/5] Loading model and tokenizer")
            model, tokenizer, config = self.load_model_and_tokenizer()

            # Step 3: Convert to CoreML
            logger.info("\n[Step 3/5] Converting to CoreML")
            logger.warning(
                "Note: This step is experimental for large LLMs. "
                "See error messages for alternative approaches if conversion fails."
            )
            mlmodel = self.convert_to_coreml(model, tokenizer)

            # Step 4: Save model
            logger.info("\n[Step 4/5] Saving model")
            self.save_model(mlmodel, tokenizer)

            # Step 5: Cleanup
            logger.info("\n[Step 5/5] Cleanup")
            del model
            gc.collect()
            logger.success("Cleanup completed")

            logger.info("\n" + "=" * 70)
            logger.success("Conversion completed successfully!")
            logger.info("=" * 70)
            logger.info(f"\nModel saved to: {self.output_dir.absolute()}")
            logger.info("\nTo use the model:")
            logger.info("  import coremltools as ct")
            logger.info(f"  model = ct.models.MLModel('{self.output_dir / 'model.mlpackage'}')")
            logger.info("\nNote: For production GPU inference on Apple Silicon,")
            logger.info("consider using MLX (pip install mlx mlx-lm) instead.")

        except KeyboardInterrupt:
            logger.warning("\nConversion interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\nConversion failed: {e}")
            logger.info("\nTroubleshooting:")
            logger.info("1. Ensure you have enough RAM (32GB+ for 30B models)")
            logger.info("2. Try using MLX instead: pip install mlx mlx-lm")
            logger.info("3. Use PyTorch MPS directly for inference")
            logger.info("4. Check coremltools documentation for LLM conversion")
            sys.exit(1)


def configure_logging(verbose: bool = True):
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

    if verbose:
        # Detailed format for verbose mode
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        # Simpler format
        log_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stdout,
        format=log_format,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
    )

    # Also log to file
    logger.add(
        "coreml_conversion.log",
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to CoreML for GPU inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Qwen3-30B for GPU
  python convert_to_coreml_gpu.py --model Qwen/Qwen3-30B-A3B --output ./models/qwen3-30b-gpu

  # Convert with FP32 (larger but more precise)
  python convert_to_coreml_gpu.py --model Qwen/Qwen3-0.6B --output ./models/qwen3 --no-fp16

  # Target CPU only
  python convert_to_coreml_gpu.py --model Qwen/Qwen3-0.6B --output ./models/qwen3 --compute-unit CPU_ONLY

Note: This script is experimental for large LLMs (30B+). For production use,
consider MLX (pip install mlx mlx-lm) or PyTorch MPS for GPU inference.
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen3-30B-A3B)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--compute-unit",
        type=str,
        default="CPU_AND_GPU",
        choices=["CPU_AND_GPU", "ALL", "CPU_ONLY"],
        help="Target compute unit (default: CPU_AND_GPU for GPU inference)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Use FP32 instead of FP16 (larger model, potentially more accurate)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for conversion (default: 512)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(verbose=not args.quiet)

    # Create converter
    converter = CoreMLGPUConverter(
        model_name=args.model,
        output_dir=args.output,
        compute_unit=args.compute_unit,
        fp16=not args.no_fp16,
        max_seq_length=args.max_seq_length,
    )

    # Run conversion
    converter.convert()


if __name__ == "__main__":
    main()
