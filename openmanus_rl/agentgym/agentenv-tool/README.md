# Agent Environments - Tool

## Apply for api key

**Movie**: https://developer.themoviedb.org/docs/getting-started

**Todo**: https://developer.todoist.com/rest/v2/#getting-started

**Sheet**: https://developers.google.cn/workspace/guides/get-started?hl=en

Then modify the last three lines of setup.sh.  Remember to put the credential.json file of the **Sheet** in agentenv-tool/ToolUsage/toolusage/utils/sheet.

Ref: https://github.com/hkust-nlp/AgentBoard/blob/main/assets/api_keys_tool.md

## Setup

``` sh
conda create -n agentenv-tool python=3.8.13
conda activate agentenv-tool
source ./setup.sh
```

## Launch
**NOTE**: launch server under `AgentEnvironments/agentenv-tool` path
``` sh
weather --host 0.0.0.0 --port 8000
movie --host 0.0.0.0 --port 8000
academia --host 0.0.0.0 --port 8000
todo --host 0.0.0.0 --port 8000
sheet --host 0.0.0.0 --port 8000
```

## Testset
```
weather: agentenv-tool/ToolUsage/data/weather.jsonl "id":0-342
movie: agentenv-tool/ToolUsage/data/movie.jsonl "id":0-237
academia: agentenv-tool/ToolUsage/data/academia.jsonl "id":0-19
todo: agentenv-tool/ToolUsage/data/todo.jsonl "id":0-154
sheet: agentenv-tool/ToolUsage/data/sheet.jsonl "id":0-19
```
