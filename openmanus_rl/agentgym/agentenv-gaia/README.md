# GAIA Environment Server

A comprehensive environment server for GAIA (Generative AI Agent) tasks, designed to work with OpenManus-RL. This server provides a research environment with powerful tools and real-time observation tracking.

## Features

- Loads GAIA datasets from HuggingFace or local files
- Integrates real functional tools for web search, Python code execution, and bash commands
- Concurrency support with environment locking
- Realistic research workflow simulation
- Detailed memory of actions and results
- Answer evaluation with true answer comparison

## Installation

```bash
pip install -e .
```

## Usage

### Starting the server

```bash
# Start the server with default settings
gaia-server

# Start with custom host and port
gaia-server --host 127.0.0.1 --port 8080

# Start with custom data directory
gaia-server --data-dir /path/to/data/
```

### API Endpoints

The GAIA environment server provides the following endpoints:

- `GET /` - Test connection
- `GET /list_envs` - List all active environments
- `POST /create` - Create a new environment
- `GET /observation?env_idx={env_id}` - Get the current observation for an environment
- `POST /step` - Execute an action in an environment
- `POST /reset` - Reset an environment
- `GET /available_actions?env_idx={env_id}` - Get available actions for an environment

### Available Tools

The environment provides the following powerful tools:

- **web_search** - Search the web for real-time information about any topic
- **bash** - Execute bash commands in the terminal
- **python_execute** - Execute Python code and get the results
- **terminate** - Submit your final answer and terminate the task

### Example API Usage

```python
import requests
import json

# Server URL
BASE_URL = "http://localhost:8000"

# Create a new environment
response = requests.post(f"{BASE_URL}/create", json={"id": 0, "dataset_type": "validation"})
env_id = response.json()
print(f"Created environment with ID: {env_id}")

# Optionally specify tools when creating an environment
response = requests.post(f"{BASE_URL}/create", 
                        json={
                            "id": 1, 
                            "dataset_type": "validation",
                            "tool_list": ["web_search", "python_execute", "terminate"]
                        })
env_id_custom_tools = response.json()

# Get initial observation
observation = requests.get(f"{BASE_URL}/observation?env_idx={env_id}").json()
print(f"Initial observation: {observation}")

# Get available actions
actions = requests.get(f"{BASE_URL}/available_actions?env_idx={env_id}").json()
print(f"Available actions: {actions}")

# Use web_search tool
search_action = "Action: web_search with Action Input: What is the capital of France?"
response = requests.post(f"{BASE_URL}/step", json={"env_idx": env_id, "action": search_action})
result = response.json()
print(f"Search results: {result['observation']}")

# Use python_execute tool
python_code = '''
def calculate_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

for i in range(10):
    print(f"Fibonacci {i}: {calculate_fibonacci(i)}")
'''
python_action = f"Action: python_execute with Action Input: {python_code}"
response = requests.post(f"{BASE_URL}/step", json={"env_idx": env_id, "action": python_action})
result = response.json()
print(f"Python execution results: {result['observation']}")

# Use bash tool
bash_action = "Action: bash with Action Input: ls -la"
response = requests.post(f"{BASE_URL}/step", json={"env_idx": env_id, "action": bash_action})
result = response.json()
print(f"Bash results: {result['observation']}")

# Submit an answer with terminate
answer = "Action: terminate with Action Input: Paris is the capital of France"
response = requests.post(f"{BASE_URL}/step", json={"env_idx": env_id, "action": answer})
result = response.json()
print(f"Final observation: {result['observation']}")
print(f"Reward: {result['reward']}")
print(f"Done: {result['done']}")

# Reset the environment
requests.post(f"{BASE_URL}/reset", json={"env_idx": env_id, "id": 1})
```

### Alternative Action Formats

The server supports multiple action formats:

#### Standard Format
```
Action: tool_name with Action Input: your_input
```

#### Direct Format
```
tool_name: your_input
```

#### JSON Format
```
{"tool_name": "web_search", "query": "What is the capital of France?"}
```

#### Specific Parameter Formats

**Web Search:**
```
Action: web_search with Action Input: your search query
```

**Python Execute:**
```
Action: python_execute with Action Input: print("Hello, World!")
```

**Bash:**
```
Action: bash with Action Input: ls -la
```

**Terminate:**
```
Action: terminate with Action Input: Your final answer text here
```

## Dataset Format

Place GAIA datasets in the `data/gaia/` directory. The server will automatically load from this location or download the datasets from HuggingFace if not available locally.

Dataset loading is handled automatically through the `load_gaia_data` utility function.
