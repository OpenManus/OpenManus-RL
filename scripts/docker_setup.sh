#!/bin/bash

# Setup script for OpenManus-RL Docker environment on AMD GPUs

set -e

echo "========================================="
echo "OpenManus-RL Docker Setup for AMD GPUs"
echo "========================================="

# Step 1: Stop and remove existing OpenManus container if it exists
echo "Cleaning up existing OpenManus container..."
docker stop openmanus-rl 2>/dev/null || true
docker rm openmanus-rl 2>/dev/null || true

# Step 2: Create a new container from the existing snapshot image
echo "Starting new OpenManus-RL container..."
docker run -it -d --name openmanus-rl \
  --ipc=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/workspace" \
  -v "/root/models:/root/models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8001:8000 \
  -w /workspace \
  verl-agent:rocm-snap1 bash

echo "Container started. Setting up environment..."

# Step 3: Install dependencies inside the container
docker exec -it openmanus-rl bash -c '
export PATH="$HOME/.local/bin:$PATH"
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)

# Create virtual environment
uv venv /opt/openmanus-venv
. /opt/openmanus-venv/bin/activate

# Install required packages
uv pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f
uv pip install -e . --no-deps
uv pip install pyyaml
uv pip install -U openai
uv pip install Ray
uv pip install together
uv pip install wikipedia python-dotenv requests

echo "Environment setup complete!"
'

echo "========================================="
echo "Setup complete! You can now:"
echo "1. Enter the container: docker exec -it openmanus-rl bash"
echo "2. Activate the environment: source /opt/openmanus-venv/bin/activate"
echo "3. Run the unified script from /workspace"
echo "========================================="

