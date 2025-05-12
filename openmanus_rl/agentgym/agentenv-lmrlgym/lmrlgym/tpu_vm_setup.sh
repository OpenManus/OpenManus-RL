install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip && python -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
# rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
