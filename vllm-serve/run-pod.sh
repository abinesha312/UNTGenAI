#!/usr/bin/env bash
set -euo pipefail

IMAGE="vllm-server:latest"

# Configuration: Set NUM_GPUS environment variable to use 4 or 8 GPUs
# Default to 8 GPUs for full H100 cluster
NUM_GPUS=${NUM_GPUS:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${NUM_GPUS}}

# Set HuggingFace cache path from environment or use default
HF_CACHE_PATH=${HF_CACHE_PATH:-"${HOME}/.cache/huggingface"}

# Calculate shared memory size based on GPU count (8GB per GPU minimum for large models)
if [ "${NUM_GPUS}" -eq 8 ]; then
    SHM_SIZE="16g"
    GPU_DEVICES="0,1,2,3,4,5,6,7"
    VISIBLE_DEVICES="0-7"
elif [ "${NUM_GPUS}" -eq 4 ]; then
    SHM_SIZE="8g"
    GPU_DEVICES="0,1,2,3"
    VISIBLE_DEVICES="0-3"
else
    echo "Error: NUM_GPUS must be 4 or 8"
    exit 1
fi

echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "  Shared Memory: ${SHM_SIZE}"
echo "  GPU Devices: ${GPU_DEVICES}"
echo ""

# 0) Clean up existing containers and pods (if they exist)
echo "Cleaning up existing containers and pods..."
podman container exists vllm && podman stop vllm && podman rm vllm || true
podman pod exists vllm-pod && podman pod stop vllm-pod && podman pod rm vllm-pod || true

# 1) build (only if missing or changed)
echo "Building container image..."
podman build -t "${IMAGE}" .

# 2) create a pod that maps port 5000
echo "Creating pod..."
podman pod create --name vllm-pod -p 5000:5000

# 3) run the container inside that pod
echo "Running container in pod with ${NUM_GPUS} GPUs..."

# Build GPU device arguments
GPU_ARGS=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ARGS="${GPU_ARGS} --device nvidia.com/gpu=${i}"
done

podman run \
  --pod vllm-pod \
  ${GPU_ARGS} \
  --security-opt=label=disable \
  --shm-size=${SHM_SIZE} \
  -d --name vllm \
  -v "${HF_CACHE_PATH}:/root/.cache/huggingface:Z" \
  -e NVIDIA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
  -e CUDA_VISIBLE_DEVICES=${GPU_DEVICES} \
  -e TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} \
  -e TORCH_COMPILE_MODE=reduce-overhead \
  -e TORCH_INDUCTOR_COMPILE_TO_EAGER=1 \
  -e TORCH_DYNAMO_DISABLE=1 \
  -e TORCH_CUDNN_V8_API_DISABLED=1 \
  -e NCCL_DEBUG=INFO \
  "${IMAGE}"

echo ""
echo "vLLM server is running in pod with ${NUM_GPUS} GPUs (TP=${TENSOR_PARALLEL_SIZE})"
echo "Check status with: podman ps"
echo "View logs with:    podman logs -f vllm"
echo "Test server with:  curl http://localhost:5000/v1/models"
echo ""
echo "To run with 4 GPUs instead: NUM_GPUS=4 ./run-pod.sh"
echo "To run with 8 GPUs (default): NUM_GPUS=8 ./run-pod.sh"
