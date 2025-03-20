# Agent Environments - SQLGym

## Setup

``` sh
conda create --name agentenv-sqlgym
conda activate agentenv-sqlgym
conda install python
cd AgentEnvironments/agentenv-sqlgym
pip install .
bash setup.sh
# or pip install -e . ？
```

## Launch

``` sh
AGENTENV_SQLGYM_BIRD_PATH=./bird sqlgym --host 0.0.0.0 --port 36002 # setup.sh will show `bird_path`
```

## Item ID

| Item ID      | Description        |
| ------------ | ------------------ |
| 0 ~ 9427     | Train set for BIRD |
| 9428 ~ 10961 | Dev set for BIRD   |
