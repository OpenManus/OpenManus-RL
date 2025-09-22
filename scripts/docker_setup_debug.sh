#!/bin/bash

# Setup script for OpenManus-RL Docker environment on AMD GPUs

set -e

echo "========================================="
echo "OpenManus-RL Docker Setup for AMD GPUs"
echo "========================================="

# Step 1: Stop and remove existing OpenManus container if it exists
echo "Cleaning up existing OpenManus container..."
docker stop openmanus-debugger 2>/dev/null || true
docker rm openmanus-debugger 2>/dev/null || true

# Step 2: Create a new container from the existing snapshot image
echo "Starting new OpenManus-RL container..."
docker run -it -d --name openmanus-debugger \
  --ipc=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/workspace" \
  -v "/root/models:/root/models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p 8003:8000 \
  -w /workspace \
  verl-agent:rocm-snap1 bash

echo "Container started. Setting up environment..."

# Step 3: Install dependencies inside the container
docker exec -it openmanus-debugger bash -c '
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

echo "setup Webshop env"
uv pip install blinker
uv pip install pyserini
cd openmanus_rl/environments/env_package/webshop/webshop
uv pip install -r requirements.txt

mkdir -p data
cd data
gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib; # items_shuffle_1000 - product scraped info
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu; # items_ins_v2_1000 - product attributes
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB; # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi; # items_ins_v2
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O # items_human_ins
cd ..

python -m ensurepip --upgrade || (curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python /tmp/get-pip.py)

# 升级基础打包工具
python -m pip install -U pip setuptools wheel

python -m pip install -U "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
python -m pip install -U "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

sudo apt-get update
sudo apt-get install -y openjdk-21-jdk
sudo apt install libopenblas-dev libomp-dev cmake -y

uv pip install faiss-cpu
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh

echo "Environment setup complete!"
'

echo "========================================="
echo "Setup complete! You can now:"
echo "1. Enter the container: docker exec -it openmanus-debugger bash"
echo "2. Activate the environment: source /opt/openmanus-venv/bin/activate"
echo "3. Run the unified script from /workspace"
echo "========================================="

