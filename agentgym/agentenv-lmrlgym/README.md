# Agent Environments - LMRL-Gym

## Setup

``` sh
git submodule init && git submodule update
conda env create --name agentenv-lmrlgym -f ./lmrlgym/environment.yml
conda activate agentenv-lmrlgym
bash ./setup.sh
```

## Launch

``` sh
lmrlgym --host 0.0.0.0 --port 36001
```

Note: When using make or wordle, your base URL should be `http://127.0.0.1:<port>/maze/` or `http://127.0.0.1:<port>/wordle/`.
