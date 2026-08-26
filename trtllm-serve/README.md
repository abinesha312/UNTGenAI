# TensorRT-LLM Serve (Optional)

This directory contains scripts and documentation for serving Gemma 3 27B using TensorRT-LLM as an alternative to vLLM.

## Prerequisites

- NVIDIA GPU with compute capability 8.0+ (e.g., H100, A100)
- TensorRT-LLM installed (see installation below)
- CUDA 12.x
- Docker (optional, for containerized deployment)

## Installation

```bash
# Install TensorRT-LLM
pip install tensorrt-llm>=0.5.0

# Or build from source for latest features
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout main
pip install -e .
```

## Building the TensorRT Engine

TensorRT-LLM requires building an optimized engine from the Gemma model weights:

```bash
# 1. Download Gemma 3 27B model from HuggingFace
huggingface-cli download google/gemma-3-27b-it --local-dir ./models/gemma-3-27b-it

# 2. Convert model to TensorRT-LLM checkpoint format
python convert_checkpoint.py \
    --model_dir ./models/gemma-3-27b-it \
    --output_dir ./trt_engines/gemma-3-27b-it/checkpoint \
    --dtype float16 \
    --tp_size 8

# 3. Build TensorRT engine
trtllm-build \
    --checkpoint_dir ./trt_engines/gemma-3-27b-it/checkpoint \
    --output_dir ./trt_engines/gemma-3-27b-it/engine \
    --gemma_version v3 \
    --max_batch_size 128 \
    --max_input_len 4096 \
    --max_output_len 512 \
    --max_beam_width 1 \
    --builder_opt 4 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --use_weight_only \
    --weight_only_precision int8
```

Note: Building the engine can take 30-60 minutes and requires ~60GB of free disk space.

## Running the TensorRT-LLM Server

### Option 1: Using Python Script

```bash
python run_trtllm_server.py \
    --engine_dir ./trt_engines/gemma-3-27b-it/engine \
    --tokenizer_dir ./models/gemma-3-27b-it \
    --host 0.0.0.0 \
    --port 5000 \
    --tp_size 8
```

### Option 2: Using OpenAI-Compatible API Server

```bash
python -m tensorrt_llm.hlapi.llm_api \
    --model_dir ./trt_engines/gemma-3-27b-it/engine \
    --tokenizer ./models/gemma-3-27b-it \
    --host 0.0.0.0 \
    --port 5000
```

## Configuration

To use TensorRT-LLM with the Chainlit app, set the environment variable:

```bash
export INFERENCE_BACKEND=trtllm
export INFERENCE_SERVER_URL=http://localhost:5000/v1
```

The Chainlit app will connect to the TensorRT-LLM server using the OpenAI-compatible API.

## Performance Comparison (Expected)

| Backend       | Throughput (tokens/s) | Latency (ms) | Memory (GB) |
|---------------|----------------------|--------------|-------------|
| vLLM          | ~1200                | ~80          | ~52         |
| TensorRT-LLM  | ~1800-2000           | ~50-60       | ~48         |

*Note: These are estimated values. Actual performance depends on hardware, model configuration, and workload.*

## Troubleshooting

### Build Failures

If engine build fails with OOM errors:
- Reduce `--builder_opt` to 2 or 3
- Use `--use_weight_only` with `int4` instead of `int8`
- Build with smaller `--max_batch_size`

### Runtime Issues

If server fails to start:
- Check CUDA version compatibility: `nvidia-smi`
- Verify engine was built for the correct GPU architecture
- Ensure sufficient GPU memory is available

## Status

**TensorRT-LLM support is optional and not required for basic functionality.**

This implementation has not been tested on the UNT H100 cluster. The vLLM backend is the primary and tested inference method. TensorRT-LLM is provided as an advanced optimization option for users who want maximum performance and are willing to manage the additional build complexity.

## References

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Gemma Model Card](https://huggingface.co/google/gemma-3-27b-it)
