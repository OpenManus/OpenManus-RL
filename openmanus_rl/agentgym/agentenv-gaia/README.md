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

### Alternative Action Formats

The server supports multiple action formats:

#### Standard Format
```
Action: tool_name Action Input: your_input
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
Action: web_search Action Input: your search query
```

**Python Execute:**
```
Action: python_execute Action Input: print("Hello, World!")
```

**Bash:**
```
Action: bash Action Input: ls -la
```

**Terminate:**
```
Action: terminate Action Input: Your final answer text here
```

## Dataset Format

Place GAIA datasets in the `data/gaia/` directory. The server will automatically load from this location or download the datasets from HuggingFace if not available locally.

Dataset loading is handled automatically through the `load_gaia_data` utility function.
