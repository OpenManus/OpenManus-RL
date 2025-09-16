# Rollout Guide (AlfWorld, GAIA, WebShop)

This guide shows how to run rollouts for the three environments using a single unified script. The script supports both OpenAI API and local OpenAI‑compatible endpoints (e.g., vLLM).

## Prerequisites

- Python venv prepared via Docker setup (see DOCKER_SETUP.md)
- .env at repo root (auto‑loaded) for API keys:
  - `OPENAI_API_KEY` for OpenAI
  - Optional tool keys (e.g., GAIA Google tools): `GOOGLE_API_KEY`, `GOOGLE_CX`
- For local inference (vLLM), start the server first (see DOCKER_SETUP.md or `serve_model.sh`).

## Unified Script

- Entry: `python scripts/rollout/unified_rollout.py`
- Core flags:
  - `--env {alfworld,gaia,webshop}` choose environment
  - `--model <name>` model name (OpenAI or local)
  - `--base_url <url>` set when using local server (e.g., `http://127.0.0.1:8000/v1`)
  - `--batch_size`, `--total_envs`, `--max_steps`, `--concurrency`
  - `--dump_path <jsonl>` save trajectories
  - `--chat_root <dir>` save chat histories under `trajectories/<ts>/<env>/<model>/`
  - `--dry_run` plan batches without creating envs/calling models
  - `--unique_envs` ensure unique task/game sampling where supported

## GAIA

Data path default: `data/gaia/val.json`

- Dry‑run (no model calls):
  - `python scripts/rollout/openmanus_rollout.py --env gaia --batch_size 2 --total_envs 4 --dry_run`

- OpenAI small run (minimal tools):
  - `python scripts/rollout/openmanus_rollout.py \
    --env gaia --model gpt-4o \
    --gaia_tools python_code_generator \
    --batch_size 1 --total_envs 2 --max_steps 30 --concurrency 2 \
    --dump_path logs/gaia/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

- Local vLLM small run:
  - `python scripts/rollout/openmanus_rollout.py \
    --env gaia --model qwen2.5-7b-alfworld --base_url http://127.0.0.1:8000/v1 \
    --gaia_tools python_code_generator \
    --batch_size 1 --total_envs 2 --max_steps 30 --concurrency 2 \
    --dump_path logs/gaia/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

## AlfWorld

Make sure AlfWorld is installed and game data downloaded (`alfworld-download -f`).

- Dry‑run (unique game files sampling):
  - `python scripts/rollout/unified_rollout.py --env alfworld --unique_envs --batch_size 2 --total_envs 4 --dry_run`

- OpenAI small run:
  - `python scripts/rollout/openmanus_rollout.py \
    --env alfworld --model gpt-4o \
    --batch_size 1 --total_envs 2 --max_steps 30 --concurrency 2 \
    --dump_path logs/alfworld/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

- Local vLLM small run:
  - `python scripts/rollout/openmanus_rollout.py \
    --env alfworld --model qwen2.5-7b-alfworld --base_url http://127.0.0.1:8000/v1 \
    --batch_size 1 --total_envs 2 --max_steps 20 --concurrency 2 \
    --dump_path logs/alfworld/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

## WebShop (optional)

To run WebShop, follow data/index setup in DOCKER_SETUP.md, then use:

- Dry‑run:
  - `python scripts/rollout/openmanus_rollout.py --env webshop --batch_size 2 --total_envs 4 --dry_run`

- OpenAI:
  - `python scripts/rollout/openmanus_rollout.py \
    --env webshop --model gpt-4o \
    --batch_size 2 --total_envs 4 --max_steps 30 --concurrency 2 \
    --dump_path logs/webshop/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

- Local vLLM:
  - `python scripts/rollout/openmanus_rollout.py \
    --env webshop --model qwen2.5-7b-alfworld --base_url http://127.0.0.1:8000/v1 \
    --batch_size 2 --total_envs 4 --max_steps 30 --concurrency 2 \
    --dump_path logs/webshop/trajectory_$(date +%Y%m%d_%H%M%S).jsonl --chat_root .`

## Outputs

- Logs: `logs/<env>/unified_run_*.log`
- Trajectory: `--dump_path` JSONL
- Chats: `trajectories/<timestamp>/<env>/<model>/` when `--chat_root` is set

