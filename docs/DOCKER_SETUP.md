# OpenManus-RL Docker Setup for AMD GPUs

This setup allows you to run OpenManus-RL alfworld rollouts in a Docker container without affecting your existing verl-agent environment.

## Prerequisites

- Docker installed and running
- AMD GPU with ROCm support
- The `verl-agent:rocm-snap1` Docker image (from your previous verl-agent setup)
- Models stored in `/root/models/`

## Setup Instructions

### 1. Initial Setup

First, run the setup script to create and configure the Docker container:

```bash
cd /root/OpenManus-RL
./scripts/docker_setup.sh
```

This will:
- Create a new Docker container named `openmanus-rl`
- Install all required dependencies
- Set up a virtual environment at `/opt/openmanus-venv`
- Port 8001 on host will map to 8000 in container (to avoid conflict with verl-agent)

### 2. Start/Access the Container

If you need to enter the container manually:

```bash
docker exec -it openmanus-rl bash
source /opt/openmanus-venv/bin/activate
cd /workspace
```

Then you can run commands directly.

### 3. Run Rollouts (Unified Script)

See ROLLOUT_GUIDE.md for detailed examples. A few quick starters:

- GAIA dry‑run:
  - `python scripts/rollout/unified_rollout.py --env gaia --batch_size 2 --total_envs 4 --dry_run`

- AlfWorld small run (OpenAI):
  - `python scripts/rollout/unified_rollout.py --env alfworld --model gpt-4o-mini --batch_size 1 --total_envs 2 --max_steps 20 --dump_path logs/alfworld/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

- GAIA small run (local vLLM):
  - `./scripts/serve_model.sh` (in another shell)
  - `python scripts/rollout/unified_rollout.py --env gaia --model qwen2.5-7b-alfworld --base_url http://127.0.0.1:8000/v1 --gaia_tools python_code_generator --batch_size 1 --total_envs 2 --max_steps 30 --dump_path logs/gaia/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

### 4. Running GAIA (Tool-Use) Rollouts

GAIA uses the tool-use environment and the dataset in `data/gaia/val.json`. Some tools need extra API keys.

Required packages for common tools are already listed in `requirements_docker.txt` (requests, python-dotenv, wikipedia). For Google search, set:

```bash
export GOOGLE_API_KEY=your-google-api-key
export GOOGLE_CX=your-custom-search-engine-id
```

There are two ways to run GAIA:

Use the unified script. Examples:

1) OpenAI API
```bash
export OPENAI_API_KEY="your-openai-api-key"
python scripts/rollout/unified_rollout.py \
  --env gaia --model gpt-4o-mini \
  --gaia_tools python_code_generator \
  --total_envs 50 --batch_size 10 --max_steps 30 --concurrency 8 \
  --dump_path logs/gaia/trajectory_$(date +%Y%m%d_%H%M%S).jsonl \
  --chat_root /workspace
```

2) Local model via vLLM (OpenAI-compatible)

First start the vLLM server (see above), then:
```bash
python scripts/rollout/unified_rollout.py \
  --env gaia --model qwen2.5-7b-alfworld --base_url http://127.0.0.1:8000/v1 \
  --gaia_tools python_code_generator \
  --total_envs 50 --batch_size 10 --max_steps 30 --concurrency 8 \
  --dump_path logs/gaia/trajectory_$(date +%Y%m%d_%H%M%S).jsonl \
  --chat_root /workspace
```

Notes:
- Default GAIA tools used in examples: `python_code_generator`（避免外部 API 依赖）。
- If a tool needs external access (web APIs), ensure the container has outbound network connectivity and env vars are set.
- Chat histories and logs are saved under `logs/gaia` and `trajectories/<timestamp>/gaia/<model>/` when `--chat_root` is provided.

## Container Management

### Stop the container
```bash
docker stop openmanus-rl
```

### Start the container again
```bash
docker start openmanus-rl
```

### Remove the container
```bash
docker stop openmanus-rl
docker rm openmanus-rl
```

### Check container logs
```bash
docker logs openmanus-rl
```

## Troubleshooting

### If vLLM fails to start
1. Check GPU memory usage: `rocm-smi`
2. Adjust `--gpu-memory-utilization` in `serve_model.sh`
3. Make sure no other process is using port 8000 in the container

### If rollout fails
1. Check that all dependencies are installed: `pip list`
2. Verify AlfWorld data is downloaded: `ls ~/.cache/alfworld` or re‑run `alfworld-download -f`
3. Check logs under `/workspace/logs/<env>/`

### Port conflicts
- Default: container 8000 → host 8001 (configured by `docker_setup.sh`)
- Adjust mapping via `-p` flag if needed.

## Output Files

- Trajectory files: `/root/OpenManus-RL/logs/alfworld/trajectory_*.jsonl`
- Chat histories: `/root/OpenManus-RL/trajectories/<timestamp>/`
- Log files: `/root/OpenManus-RL/logs/alfworld/run_log_*.log`
