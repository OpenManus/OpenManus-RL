# Evaluation Guide

This guide explains how to quickly set up and run WebShop environment evaluations using vLLM for accelerated throughput, as well as fallback options for the classic (slower) evaluation scripts.

---

## Prerequisites

1. **Python 3.8+** installed on your system.
2. **pip** package manager.
3. **vLLM Server** (OpenAI-compatible) or local vLLM endpoint.
4. **AgentENV–WebShop** service.

---

## 1. Install and Prepare the Package

From the root of the repository:

```bash
# Install the agentgym package in editable mode
cd openmanus_rl/agentgym/agentenv/
pip install -e .
```

This ensures `agentenv` is available on your `PYTHONPATH` without extra hacks.

---

## 2. Start the WebShop Environment Server

1. Navigate to the WebShop service directory:

   ```bash
   cd agentenv/agentenv-webshop
   ```

2. Launch the server on port **36001** (or your preferred port):

   ```bash
   webshop --host 0.0.0.0 --port 36001
   ```

Leave this process running in the background or in a separate terminal.

---

## 3. Run vLLM‑Accelerated Evaluation

1. Execute the helper script to start or configure your vLLM server:

   ```bash
   bash openmanus_rl/evaluation/run_vllm.sh
   ```

2. Ensure your `PYTHONPATH` includes the `agentenv` package and you are under the OpenManus-RL/openmanus_rl directory:

   ```bash
   export PYTHONPATH=./agentgym/agentenv:$PYTHONPATH
   ```

3. Launch the evaluation driver:

   ```bash
   python openmanus_rl/evaluation/vllm_eval_webshop.py
   ```

This will run tasks against the WebShop environment via your vLLM endpoint for maximum speed.

---

## 4. Legacy Evaluation Scripts (Slower)

If you prefer the classic evaluation (single‑model, non‑vLLM), use one of these scripts:

* **Basic (single‑threaded):**

  ```bash
  bash agentgym/agentenv/examples/basic/base_eval_webshop.sh
  ```

* **Distributed (multi‑worker):**

  ```bash
  bash agentgym/agentenv/examples/distributed_eval_scripts/distributed_eval_webshop.sh
  ```

Expect these to run significantly slower than the vLLM‑driven workflow.

---

## 5. Troubleshooting

* **ModuleNotFoundError:**
  Make sure you ran `pip install -e .` in `openmanus_rl/agentgym` and removed any stale `PYTHONPATH` overrides.

* **Port conflicts on 36001:**
  Either kill the process using that port or choose a different port and update both the WebShop server and evaluation script arguments.

* **vLLM connection errors:**
  Verify that your vLLM server is up (`run_vllm.sh` logs) and that the `base_url` in your script matches its endpoint.

---

## License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.
