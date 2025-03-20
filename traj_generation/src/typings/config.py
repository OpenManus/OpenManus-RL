import datetime
import json
import sys
from typing import Dict, List, Union

from pydantic import BaseModel, validator

from src.utils import ColorMessage
from .general import InstanceFactory, Assignment


class ConcurrencyConfig(BaseModel):
    agent: Dict[str, int]
    task: Dict[str, int]


class DefinitionConfig(BaseModel):
    agent: Dict[str, InstanceFactory]
    task: Dict[str, InstanceFactory]


def get_predefined_structure():
    now = datetime.datetime.now()
    return {
        "TIMESTAMP": now.strftime("%Y-%m-%d-%H-%M-%S"),
        "TIMESTAMP_DATE": now.strftime("%Y-%m-%d"),
        "TIMESTAMP_TIME": now.strftime("%H-%M-%S"),
    }


class AssignmentConfig(BaseModel):
    assignments: List[Assignment]
    concurrency: ConcurrencyConfig
    definition: DefinitionConfig
    output: str = None

    @validator("assignments", pre=True)
    def assignments_validation(cls, v):
        assert isinstance(v, list), f"'assignments' must be a list, but got {type(v)}"
        ret = []
        for item in v:
            assert isinstance(
                item, dict
            ), f"Each item in 'assignments' must be a dict, but got {type(item)}"
            agent = item.get("agent", None)
            if agent is None:
                raise ValueError("'agent' must be specified")
            if isinstance(agent, str):
                agent = [agent]
            task = item.get("task")
            if task is None:
                raise ValueError("'task' must be specified")
            if isinstance(task, str):
                task = [task]
            for a in agent:
                for t in task:
                    ret.append(Assignment(agent=a, task=t))
        return ret

    @validator("output", pre=True)
    def output_validation(cls, v):
        predefined_structure = get_predefined_structure()
        if v is None:
            v = "output/{TIMESTAMP}"
        assert isinstance(v, str), f"'output' must be a string, but got {type(v)}"
        return v.format(**predefined_structure)

    @classmethod
    def post_validate(cls, instance: "AssignmentConfig"):

        REMOVE_UNUSED_IN_DEFINITION = True
        REMOVE_UNUSED_IN_CONCURRENCY = True

        # Step 1. Check if all agents and tasks are defined, and remove unused agents and tasks

        agent_in_assignment = set()
        task_in_assignment = set()
        for assignment in instance.assignments:
            assert (
                assignment.agent in instance.definition.agent
            ), f"Agent {assignment.agent} is not defined."
            agent_in_assignment.add(assignment.agent)
            assert (
                assignment.task in instance.definition.task
            ), f"Task {assignment.task} is not defined."
            task_in_assignment.add(assignment.task)

        for agent in agent_in_assignment:
            assert (
                agent in instance.concurrency.agent
            ), f"Concurrency of {agent} is not specified."
        for task in task_in_assignment:
            assert (
                task in instance.concurrency.task
            ), f"Concurrency of {task} is not specified."

        def remove_unused(
            target: Union[DefinitionConfig, ConcurrencyConfig], warning_suffix: str
        ):
            nonlocal agent_in_assignment, task_in_assignment
            removed_agents = set()
            removed_tasks = set()

            for definition_agent in target.agent.keys():
                if definition_agent not in agent_in_assignment:
                    removed_agents.add(definition_agent)

            for definition_task in target.task.keys():
                if definition_task not in task_in_assignment:
                    removed_tasks.add(definition_task)

            if len(removed_agents) > 0 or len(removed_tasks) > 0:

                print(
                    ColorMessage.yellow(
                        f"Warning: {len(removed_agents)} agent(s) and {len(removed_tasks)} task(s) are "
                        + warning_suffix
                    ),
                    file=sys.stderr,
                )
                print(ColorMessage.yellow(f"    Agent: {removed_agents}"))
                print(ColorMessage.yellow(f"    Task: {removed_tasks}"))

                for agent in removed_agents:
                    target.agent.pop(agent)

                for task in removed_tasks:
                    target.task.pop(task)

        if REMOVE_UNUSED_IN_DEFINITION:
            remove_unused(
                instance.definition, "defined but not used, they will be ignored."
            )

        if REMOVE_UNUSED_IN_CONCURRENCY:
            remove_unused(
                instance.concurrency,
                "specified in concurrency but not defined, they will be ignored.",
            )

        # Step 2. remove duplicated assignments

        assignments = set()
        for assignment in instance.assignments:
            target = (assignment.agent, assignment.task)
            if target in assignments:
                agent_ = json.dumps(target[0], ensure_ascii=False)
                task_ = json.dumps(target[1], ensure_ascii=False)
                print(
                    ColorMessage.yellow(
                        f"Warning: Assignment(agent={agent_}, task={task_}) "
                        f"is duplicated, only the first one will be kept."
                    )
                )
            assignments.add(target)
        instance.assignments = []
        for agent, task in assignments:
            instance.assignments.append(Assignment(agent=agent, task=task))

        return instance
