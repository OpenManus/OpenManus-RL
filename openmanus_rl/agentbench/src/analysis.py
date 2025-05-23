import argparse
import datetime
import json
import os
import re
import time
from collections import OrderedDict

import yaml

from .configs import ConfigLoader
from .utils import ColorMessage

MODEL_MAP = {
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo-0613": "gpt-3.5-turbo",
    "llama-2-13b": "llama2-13b",
    "llama-2-7b": "llama2-7b",
    "chatglm-6b": "chatglm-6b",
    "wizard-30b": "wizardlm-30b",
    "vicuna-33b": "vicuna-33b",
    "oasst-12b": "oasst-12b",
    "guanaco-65b": "guanaco-65b",
    "koala-13b": "koala-13b",
    "text-davinci-003": "text-davinci-003",
    "wizard-13b": "wizardlm-13b",
    "guanaco-33b": "guanaco-33b",
    "text-davinci-002": "text-davinci-002",
    "llama2-70b": "llama2-70b",
    "codellama": "codellama-34b",
    "openchat": "openchat-13b",
    "claude-ins": "claude-instant",
    "claude-v1.3": "claude",
    "claude-2": "claude-2",
    "codellama-13b": "codellama-13b",
    "codellama-7b": "codellama-7b",
    "codegeex2-6b": "codegeex2-6b",
    "dolly": "dolly-12b",
    "vicuna-7b": "vicuna-7b",
    "vicuna-13b": "vicuna-13b",
    "chat-bison": "chat-bison-001",
}

VALIDATION_MAP_FUNC = {
    "Completed": lambda x: x["COMPLETED"],
    "Context Limit Exceeded": lambda x: x["AGENT_CONTEXT_LIMIT"],
    "Invalid Format": lambda x: x["AGENT_VALIDATION_FAILED"],
    "Invalid Action": lambda x: x["AGENT_INVALID_ACTION"],
    # Not in above list
    "Task Limit Exceeded": lambda x: sum(
        [x[t] for t in x if t in ["UNKNOWN", "TASK_ERROR", "TASK_LIMIT_REACHED"]]
    ),
}


def analyze_output(config: str, output: str, since_timestamp: float):
    """
    Walk through the output folder (including sub-dir) and analyze the overall.json file
    Rule:
        - valid overall file: **/{agent}/{task}/overall.json
        - if a same (agent, task) pair, select the latest one
    """
    loader = ConfigLoader()
    config: dict = loader.load_from(config)
    assert "definition" in config, "definition not found in config"
    assert "agent" in config["definition"], "agent not found in config.definition"
    assert "task" in config["definition"], "task not found in config.definition"
    agents = set(config["definition"]["agent"].keys()).intersection(
        set(MODEL_MAP.keys())
    )
    tasks = list(config["definition"]["task"].keys())

    print(
        ColorMessage.cyan(
            f"Available Agents ({len(agents)}):\n    "
            + "\n    ".join(agents)
            + "\n\n"
            + f"Available Tasks ({len(tasks)}):\n    "
            + "\n    ".join(tasks)
            + "\n"
        )
    )

    overall_dict = OrderedDict()  # agent -> task -> {file: str, time: float}
    for root, dirs, files in os.walk(output):
        if "overall.json" in files:
            # get full path of root
            root = os.path.abspath(root)
            # get agent and task name
            pattern = root.split("/")
            if len(pattern) < 2:
                continue
            agent = pattern[-2]
            task = pattern[-1]
            ct = os.path.getmtime(os.path.join(root, "overall.json"))
            if agent not in agents:
                continue
            elif task not in tasks:
                continue
            elif ct < since_timestamp:
                continue
            agent = MODEL_MAP[agent]
            if agent in overall_dict and task in overall_dict[agent]:
                # get time
                if ct < overall_dict[agent][task]["time"]:
                    continue
            overall_dict.setdefault(agent, OrderedDict())
            overall_dict[agent][task] = {
                "file": os.path.join(root, "overall.json"),
                "time": os.path.getmtime(os.path.join(root, "overall.json")),
            }

    # agent -> task -> {file: str, time: str(YYYY-MM-DD HH:MM:SS), overall: dict}

    agent_names = []
    task_names = []
    validation_names = []

    for agent in overall_dict:
        if agent not in agent_names:
            agent_names.append(agent)
        for task in overall_dict[agent]:
            if task not in task_names:
                task_names.append(task)
            overall_dict[agent][task]["time"] = datetime.datetime.fromtimestamp(
                overall_dict[agent][task]["time"]
            ).strftime("%Y-%m-%d %H:%M:%S")
            with open(overall_dict[agent][task]["file"], "r", encoding="utf-8") as f:
                overall_dict[agent][task]["overall"] = json.load(f)
            if "validation" in overall_dict[agent][task]["overall"]:
                overall_dict[agent][task]["overall"]["validation"] = {
                    validation: VALIDATION_MAP_FUNC[validation](
                        overall_dict[agent][task]["overall"]["validation"]
                    )
                    for validation in VALIDATION_MAP_FUNC
                }
                for validation in overall_dict[agent][task]["overall"]["validation"]:
                    if validation not in validation_names:
                        validation_names.append(validation)

    return agent_names, task_names, validation_names, overall_dict


class TaskHandler:
    def match(self, task_name) -> bool:
        raise NotImplementedError()

    def get_main_metric(self, overall_result):
        raise NotImplementedError()

    def get_order_priority(self):
        return 100000

    @staticmethod
    def get_handler(task_name) -> "TaskHandler":
        handlers = [DCG(), HH(), OS(), DB(), KG(), LTP(), WB(), WS()]
        for handler in handlers:
            if handler.match(task_name):
                return handler
        raise ValueError(f"Unknown task: {task_name}")


class DCG(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return (
            "card" in task_name
            or task_name.startswith("cg")
            or task_name.startswith("dcg")
        )

    def get_main_metric(self, overall_result):
        try:
            return overall_result["custom"]["score"]
        except:
            return {"win_rate(legacy)": overall_result["custom"]["win_rate"]}

    def get_order_priority(self):
        return 4


class HH(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("alf")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["overall"]["success_rate"]

    def get_order_priority(self):
        return 6


class OS(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("os") or task_name.startswith("operating")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["overall"]["acc"]

    def get_order_priority(self):
        return 1


class DB(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("db") or task_name.startswith("database")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["overall_cat_accuracy"]

    def get_order_priority(self):
        return 2


class KG(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("kg") or task_name.startswith("knowledge")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["main"]

    def get_order_priority(self):
        return 3


class LTP(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("ltp") or task_name.startswith("literal")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["main"]

    def get_order_priority(self):
        return 5


class WB(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("m2w") or task_name.startswith("mind2web")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["step_sr"] / 100

    def get_order_priority(self):
        return 8


class WS(TaskHandler):
    def match(self, task_name) -> bool:
        task_name = task_name.lower()
        return task_name.startswith("ws") or task_name.startswith("webshop")

    def get_main_metric(self, overall_result):
        return overall_result["custom"]["reward"]

    def get_order_priority(self):
        return 7


def parse_timestamp(time_str: str) -> float:
    # is a int or float
    try:
        return float(time_str)
    except:
        pass
    # is a datetime
    try:
        return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
    except:
        pass
    try:
        return datetime.datetime.strptime(time_str, "%Y-%m-%d").timestamp()
    except:
        pass
    try:
        return datetime.datetime.strptime(time_str, "%Y-%m").timestamp()
    except:
        pass
    # is a time delta (e.g. 1d, 1h, 1m, 1s)
    num = float(re.findall(r"[\d\.]+", time_str)[0])
    unit = re.findall(r"[a-zA-Z]+", time_str)[0]
    if unit == "d":
        delta = num * 24 * 60 * 60
    elif unit == "h":
        delta = num * 60 * 60
    elif unit == "m":
        delta = num * 60
    elif unit == "s":
        delta = num
    else:
        raise Exception("Unknown time unit")
    return time.time() - delta


def main(args):
    agent_names, task_names, validation_names, details = analyze_output(
        args.config, args.output, parse_timestamp(args.time)
    )
    task_names.sort(key=lambda x: TaskHandler.get_handler(x).get_order_priority())
    summary = OrderedDict()
    for agent in details:
        summary[agent] = OrderedDict()
        for task in details[agent]:
            handler = TaskHandler.get_handler(task)
            if handler is not None:
                summary[agent][task] = handler.get_main_metric(
                    details[agent][task]["overall"]
                )
            else:
                summary[agent][task] = details[agent][task]["overall"]

    for agent in details:
        for task in details[agent]:
            print(
                ColorMessage.cyan(
                    f"Agent: {agent:20} Task: {task:20} Path: {details[agent][task]['file']}"
                )
            )

    final_result = {
        "summary": summary,
        "details": details,
    }

    os.makedirs(args.save, exist_ok=True)

    # Overall Calculation

    with open(os.path.join(args.save, "result.json"), "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(args.save, "result.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(final_result, f, indent=4, allow_unicode=True, sort_keys=True)
    with open(os.path.join(args.save, "summary.csv"), "w", encoding="utf-8") as f:
        """
        Format:
            Agent\\Task, Task1, Task2, ...
            Agent1, MainMetric(Agent1,Task1), MainMetric(Agent1,Task2), ...
            ......
        """
        f.write("Agent\\Task," + ",".join(task_names) + "\n")
        for agent in summary:
            f.write(
                agent
                + ","
                + ",".join(
                    [
                        (str(summary[agent][task]) if task in summary[agent] else "")
                        for task in task_names
                    ]
                )
                + "\n"
            )

    # Validation Analysis
    agent_validations = {
        agent: {validation: [] for validation in validation_names}
        for agent in agent_names
    }
    task_validations = {
        task: {validation: [] for validation in validation_names} for task in task_names
    }

    for agent in summary:
        for task in summary[agent]:
            if "validation" in details[agent][task]["overall"]:
                for validation in details[agent][task]["overall"]["validation"]:
                    agent_validations[agent][validation].append(
                        details[agent][task]["overall"]["validation"][validation]
                    )
                    task_validations[task][validation].append(
                        details[agent][task]["overall"]["validation"][validation]
                    )

    # Agent-Centric Validation Analysis
    with open(
        os.path.join(args.save, "agent_validation.csv"), "w", encoding="utf-8"
    ) as f:
        """
        Format:
            Agent\\Validation, Validation1, Validation2, ...
            Agent1, Avg(Agent1,Validation1), Avg(Agent1,Validation2), ...
            ......
        """
        f.write("Agent\\Validation," + ",".join(validation_names) + "\n")
        for agent in agent_validations:
            f.write(
                agent
                + ","
                + ",".join(
                    [
                        (
                            str(
                                sum(agent_validations[agent][validation])
                                / len(agent_validations[agent][validation])
                            )
                            if validation in agent_validations[agent]
                            and len(agent_validations[agent][validation]) > 0
                            else "--"
                        )
                        for validation in validation_names
                    ]
                )
                + "\n"
            )

    # Task-Centric Validation Analysis
    with open(
        os.path.join(args.save, "task_validation.csv"), "w", encoding="utf-8"
    ) as f:
        """
        Format:
            Task\\Validation, Validation1, Validation2, ...
            Task1, Avg(Task1,Validation1), Avg(Task1,Validation2), ...
            ......
        """
        f.write("Task\\Validation," + ",".join(validation_names) + "\n")
        for task in task_validations:
            f.write(
                task
                + ","
                + ",".join(
                    [
                        (
                            str(
                                sum(task_validations[task][validation])
                                / len(task_validations[task][validation])
                            )
                            if validation in task_validations[task]
                            and len(task_validations[task][validation]) > 0
                            else "--"
                        )
                        for validation in validation_names
                    ]
                )
                + "\n"
            )

    print(ColorMessage.green(f"Analysis result saved to {os.path.abspath(args.save)}"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config", type=str, default="configs/assignments/definition.yaml"
    )
    arg_parser.add_argument("-o", "--output", type=str, default="outputs")
    arg_parser.add_argument("-s", "--save", type=str, default="analysis")
    arg_parser.add_argument("-t", "--time", type=str, default="0")
    args = arg_parser.parse_args()
    main(args)
