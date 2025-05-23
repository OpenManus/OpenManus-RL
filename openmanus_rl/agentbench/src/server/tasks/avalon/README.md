# AvalonBench

## Quick Start 

### Start the task server and the assigner

Start the game (3 is the number of workers)
```bash
python -m src.start_task -a --start avalon-dev-single 3
```
Start the assigner
```bash
python -m src.assigner --config ./configs/assignments/test_avalon.yaml
```

### Customize configurations and data

1. You can modify the file `configs/tasks/avalon.yaml` to configure the agent list. A config file looks like this:
```yaml
default:
  module: "src.server.tasks.avalon.AvalonBench"
  parameters:
    num_players: 5
    discussion: False

avalon-dev-naive:
  parameters:
    name: "AvalonBench-dev-naive"
    data_file: "data/avalon/dev.json"
    agent_list: ["naive", "naive", "naive", "naive", "naive"]

avalon-dev-single:
  parameters:
    name: "AvalonBench-dev-single"
    data_file: "data/avalon/dev.json"
    agent_list: ["llm", "naive", "naive", "naive", "naive"]
```
where `naive` stands for the naive bots. Agents will play the roles with the same index in the data file (see following).
```plaintext
Note: There should only be one "llm" in the `agent_list`
```

2. You can also add data in `data/avalon/dev.json` (Note: Currently we only support the 5-player game setting, which includes 1 Merlin, 2 Servants, 1 Minion and 1 Assassin). A data item looks like this:

```json
 {
     "num_players": 5,
     "quest_leader": 0,
     "role_names": ["Assassin", "Servant", "Servant", "Merlin", "Minion"]
 }
```
where `quest_leader` is the id of the initial quest leader in this game. You can change the game setup by altering `quest_leader` with number from 0 to 4, and by permuting `role_names`.

### Naive experiment

You can also start a naive experiment using:
```bash
python -m src.start_task -a --start avalon-dev-naive 3
```
where all the agents are naive bots. For details of the naive strategies, please refer to the [paper](https://arxiv.org/pdf/2310.05036.pdf).

## Prompts

All the prompts are maintained in `src/server/tasks/avalon/prompt.py`. You can find the respective prompts used in `src/server/tasks/avalon/agents/llm_with_discussion.py` and `src/server/tasks/avalon/wrapper.py`.