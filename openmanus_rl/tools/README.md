
## GAIA web search

On Python 3.10+, GAIA/tool-use runs use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
when neither `GOOGLE_API_KEY` nor `GOOGLE_CX` is set. Search uses Fast mode and
needs no account or API key. The anonymous service has usage limits; errors are
returned to the agent without switching providers or retrying automatically.
Existing Google credentials, including credentials loaded from `.env`, keep
Google selected. Set both Google variables for a working Google configuration.

Install the search dependencies in your existing rollout environment from the
repository root (they are also declared in the main dependency manifests):

```sh
pip install -r requirements-search.txt
```

`scripts/rollout/gaia.py` and `scripts/rollout/openmanus_rollout.py --env gaia`
select this default automatically. To select it explicitly, use
`--tools parallel_search wikipedia_knowledge_searcher python_code_generator`
with `gaia.py`, or `--gaia_tools` with `openmanus_rollout.py`. Explicit tool lists
and `config.env.available_tools` are used as supplied. Omit search tools from
those lists to disable web search; an empty configuration list disables all tools.
`google_search` remains available explicitly, with no fallback after an error.
Python versions before 3.10 retain the Google default.

The agent can search or fetch page excerpts using the same tool:

```text
<tool_call>
tool: parallel_search
parameters: {"query": "2024 Nobel chemistry winners"}
</tool_call>

<tool_call>
tool: parallel_search
parameters: {"urls": ["https://www.nobelprize.org/prizes/chemistry/2024/summary/"], "query": "Who won?"}
</tool_call>
```

Results include source URLs and excerpts in the next agent observation. Each
environment reset creates a new research session; related searches and fetches
within one task share it. Requests have a 30-second total timeout. This provider
does not configure the separate modular-stage `search` wrapper.

## Testing the Tools

To test the text detection tool, follow these steps:

1. **Navigate to the Project Directory:**

   Change your current directory to where the tools are located. Replace `your_path` with the actual path to your project directory.

   ```sh
   cd your_path/openmanus_rl
   ```

2. **Run the Text Detection Tool:**

   ```sh
   export PYTHONPATH=$(pwd)
   ```


   Execute the tool using the following command:

   ```sh
   python tools/text_detector/tool.py

   python tools/object_detector/tool.py

   ```

## File Structure

The project is organized as follows:

```sh
├── __init__.py              # Initializes the tools package and possibly exposes submodules
├── base.py                  # Base class for tools, providing common functionality
├── text_detector/           # Directory for the text detection tool
│   ├── readme.md            # Documentation for the text detection tool
│   └── tool.py              # Implementation of the text detection tool
├── object_detector/         # Directory for the object detection tool
│   ├── readme.md            # Documentation for the object detection tool
│   └── tool.py              # Implementation of the object detection tool
```
