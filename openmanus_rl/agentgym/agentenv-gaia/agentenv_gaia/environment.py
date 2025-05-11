import asyncio
import json
import os
import re
import threading
import uuid
import uvicorn
# Import utilities from load_data
from agentenv_gaia.load_data import load_gaia_data, parse_tools
from agentenv_gaia.tool_manager import ToolManager
from gaia.base import ToolResult

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple, Union


# List of default tools to include in every environment
DEFAULT_TOOLS = ["web_search", "bash", "python_execute", "browser_use", "terminate"]



# GaiaEnvServer class to manage environment instances
class GaiaEnvServer:
    def __init__(self, max_envs=100):
        """Initialize the GAIA environment server"""
        self.env_instances = {}  # Dictionary to store environment instances
        self.env_locks = {}  # Locks for thread safety
        self.max_envs = max_envs

        # Default dataset path
        self.data_dir = os.path.join("data", "gaia")

        # Initialize the tool manager
        self.tool_manager = ToolManager()

        # Create event loop for async tools
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Try to load any available datasets
        self._preload_datasets()

    def _preload_datasets(self):
        """Pre-load available GAIA datasets"""
        try:
            # Check if datasets can be loaded
            self.validation_data = load_gaia_data(data_dir="data/", dataset="validation")
            self.test_data = load_gaia_data(data_dir="data/", dataset="test")
            print(f"Preloaded validation dataset with {len(self.validation_data)} items")
            print(f"Preloaded test dataset with {len(self.test_data)} items")
        except Exception as e:
            print(f"Warning: Could not preload datasets: {e}")
            self.validation_data = None
            self.test_data = None

    def create(self, id: int = 0, dataset_type: str = "validation", tool_list: Optional[List[str]] = None):
        """
        Create a new environment instance

        Args:
            id: Task ID in the dataset
            dataset_type: Dataset type (validation/test)
            tool_list: List of specific tools to enable

        Returns:
            str: New environment UUID
        """
        if len(self.env_instances) >= self.max_envs:
            raise ValueError(f"Maximum number of environments reached ({self.max_envs})")

        # Generate a unique environment ID
        env_id = str(uuid.uuid4())

        # Create a lock for this environment
        self.env_locks[env_id] = threading.Lock()

        # Prepare the dataset item
        dataset_item = self._prepare_dataset_item(id, dataset_type)

        # Determine available tools based on task and tool_list
        available_tools = self._determine_tools(dataset_item, tool_list)

        # Create environment with the dataset and tools
        with self.env_locks[env_id]:
            self.env_instances[env_id] = {
                "dataset": dataset_item,
                "question": dataset_item.get("question", ""),
                "state": {
                    "steps_taken": 0,
                    "done": False,
                    "reward": 0.0,
                    "goal": dataset_item.get("question", ""),
                    "context": dataset_item.get("Context", ""),
                    "files": dataset_item.get("file_name", ""),
                    "memory": [],  # Store previous actions and results
                    "collected_info": []  # Store information collected during the session
                },
                "available_tools": available_tools
            }

        return env_id

    def _prepare_dataset_item(self, id: int, dataset_type: str = "validation"):
        """
        Prepare dataset item for the given ID

        Args:
            id: Task ID
            dataset_type: Dataset type

        Returns:
            dict: Dataset item
        """
        try:
            # Try to get data from preloaded dataset
            if dataset_type == "validation" and self.validation_data is not None:
                if 0 <= id < len(self.validation_data):
                    return self.validation_data.iloc[id].to_dict()

            if dataset_type == "test" and self.test_data is not None:
                if 0 <= id < len(self.test_data):
                    return self.test_data.iloc[id].to_dict()

            # Fall back to direct loading if needed
            data = load_gaia_data(data_dir="data/", dataset=dataset_type)
            if 0 <= id < len(data):
                return data.iloc[id].to_dict()
        except Exception as e:
            print(f"Error loading dataset: {e}")

        # Return a default dataset structure if all else fails
        return {
            "question": f"Task {id} from {dataset_type} dataset",
            "Context": "No context available",
            "file_name": "",
            "task": "generic",
        }

    def _determine_tools(self, dataset_item: Dict[str, Any], tool_list: Optional[List[str]] = None) -> List[str]:
        """
        Determine tools available for this environment based on the task

        Args:
            dataset_item: Dataset item
            tool_list: Optional list of specific tools to enable

        Returns:
            List[str]: List of available tool names
        """
        # If tool_list is provided, use only those tools
        if tool_list:
            # Filter to only valid tools
            available_tools = [
                tool for tool in tool_list
                if tool in self.tool_manager.get_tool_names()
            ]

            # Always include terminate
            if "terminate" not in available_tools:
                available_tools.append("terminate")

            return available_tools

        # Get default tools
        available_tools = DEFAULT_TOOLS.copy()

        # Task-specific additions could go here
        task_type = dataset_item.get("task", "").lower()

        # Add tools based on task requirements if needed
        # This would typically come from dataset metadata

        return available_tools

    def reset(self, env_id: str, id: Optional[int] = None, dataset_type: str = "validation"):
        """
        Reset environment

        Args:
            env_id: Environment ID
            id: Optional new task ID
            dataset_type: Dataset type

        Returns:
            None
        """
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            env = self.env_instances[env_id]

            if id is not None:
                # Reset with a new task
                dataset_item = self._prepare_dataset_item(id, dataset_type)
                env["dataset"] = dataset_item
                env["question"] = dataset_item.get("question", "")

                # Determine available tools based on the new task
                env["available_tools"] = self._determine_tools(dataset_item)

                env["state"] = {
                    "steps_taken": 0,
                    "done": False,
                    "reward": 0.0,
                    "goal": dataset_item.get("question", ""),
                    "context": dataset_item.get("Context", ""),
                    "files": dataset_item.get("file_name", ""),
                    "memory": [],
                    "collected_info": []
                }
            else:
                # Reset with the same task
                dataset_item = env["dataset"]
                env["state"] = {
                    "steps_taken": 0,
                    "done": False,
                    "reward": 0.0,
                    "goal": dataset_item.get("question", ""),
                    "context": dataset_item.get("Context", ""),
                    "files": dataset_item.get("file_name", ""),
                    "memory": [],
                    "collected_info": []
                }

    def step(self, env_id: str, action: str):
        """
        Execute environment step

        Args:
            env_id: Environment ID
            action: Action description

        Returns:
            tuple: (observation, reward, done, info)
        """
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            env = self.env_instances[env_id]

            # Parse and process the action
            action_type, action_input = self._parse_action(action)

            # Update state
            env["state"]["steps_taken"] += 1

            # Process the action using available tools
            observation, reward, done = self._process_action(env_id, action_type, action_input)

            # Store action and result in memory
            env["state"]["memory"].append({
                "action": action,
                "result": observation,
                "step": env["state"]["steps_taken"]
            })

            # Update state with results
            env["state"]["reward"] += reward
            env["state"]["done"] = done

            info = {
                "steps_taken": env["state"]["steps_taken"],
                "action_processed": action,
            }

            return observation, reward, done, info

    def _parse_action(self, action: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse the action string into type and input

        Args:
            action: Action string

        Returns:
            tuple: (action_type, action_parameters)
        """
        try:
            # Parse actions in format "Action: action_type with Action Input: action_input"
            if "Action:" in action and "Action Input:" in action:
                action_parts = action.split("Action Input:", 1)
                action_type = action_parts[0].replace("Action:", "").strip()
                action_input = action_parts[1].strip()
                return action_type, {"query": action_input} if action_type == "web_search" else {
                    "command": action_input} if action_type == "bash" else {
                    "code": action_input} if action_type == "python_execute" else {"input": action_input}

            # Parse JSON format with tool_name and parameters
            elif action.strip().startswith("{") and action.strip().endswith("}"):
                try:
                    action_json = json.loads(action)
                    if "tool_name" in action_json:
                        tool_name = action_json.pop("tool_name")
                        return tool_name, action_json
                except json.JSONDecodeError:
                    pass

            # Parse direct tool calls (e.g., "web_search: query")
            elif ":" in action:
                tool_name, input_text = action.split(":", 1)
                tool_name = tool_name.strip()
                input_text = input_text.strip()

                # Map to appropriate parameter name based on tool
                if tool_name.lower() in ["web_search", "search_web"]:
                    return tool_name, {"query": input_text}
                elif tool_name.lower() == "bash":
                    return tool_name, {"command": input_text}
                elif tool_name.lower() == "python_execute":
                    return tool_name, {"code": input_text}
                elif tool_name.lower() == "browser_use":
                    return tool_name, {"url": input_text}
                elif tool_name.lower() == "terminate":
                    return "terminate", {"answer": input_text}
                else:
                    return tool_name, {"input": input_text}

            # Handle plain text as a default case
            return "unknown", {"input": action}

        except Exception as e:
            print(f"Error parsing action: {e}")
            return "unknown", {"input": action}

    def _process_action(self, env_id: str, action_type: str, action_input: Dict[str, Any]) -> Tuple[str, float, bool]:
        """
        Process the action using available tools

        Args:
            env_id: Environment ID
            action_type: Type of action
            action_input: Input parameters for the action

        Returns:
            tuple: (observation, reward, done)
        """
        env = self.env_instances[env_id]
        available_tools = env["available_tools"]

        # Clean up action type
        clean_action_type = action_type.lower().replace(" ", "_")
        if clean_action_type == "search_web":
            clean_action_type = "web_search"  # Normalize to official name

        # Check if the tool is available for this environment
        if clean_action_type in available_tools:
            try:
                # Handle terminate as a special case
                if clean_action_type == "terminate":
                    answer = action_input.get("answer", "") or action_input.get("input", "")
                    return self._process_answer_submission(env_id, answer)

                # Execute the tool using the tool manager
                tool_result = self.loop.run_until_complete(
                    self.tool_manager.execute_tool(clean_action_type, **action_input)
                )

                # Extract result
                if isinstance(tool_result, dict):
                    result_text = tool_result.get("observation", str(tool_result))
                elif isinstance(tool_result, ToolResult):
                    result_text = str(tool_result)
                else:
                    result_text = str(tool_result)

                # Add to collected info
                if result_text and not env["state"]["done"]:
                    env["state"]["collected_info"].append(result_text)

                observation = f"Tool {clean_action_type} executed.\nResult: {result_text}"
                reward = 0.1  # Small reward for successful tool use
                done = False

                return observation, reward, done

            except Exception as e:
                observation = f"Error executing tool {clean_action_type}: {str(e)}"
                reward = -0.1  # Small penalty for error
                done = False
                return observation, reward, done
        else:
            observation = f"Tool '{action_type}' is not available for this task. Available tools: {', '.join(available_tools)}"
            reward = -0.05  # Very small penalty for using unavailable tool
            done = False
            return observation, reward, done

    def _process_answer_submission(self, env_id: str, answer: str) -> Tuple[str, float, bool]:
        """
        Process answer submission

        Args:
            env_id: Environment ID
            answer: Submitted answer

        Returns:
            tuple: (observation, reward, done)
        """
        env = self.env_instances[env_id]

        # Get the true answer if available
        dataset_item = env["dataset"]
        true_answer = dataset_item.get("true_answer", None)

        # Check if this is a plausible answer based on collected info
        has_relevant_info = len(env["state"]["collected_info"]) > 0

        # Default quality is medium
        quality_reward = 0.5

        # If we have a true answer, check if the answer matches
        if true_answer and isinstance(true_answer, str):
            # Very simple heuristic - check if key terms from true answer are in the submission
            key_terms = set(re.findall(r'\b\w+\b', true_answer.lower()))
            submission_terms = set(re.findall(r'\b\w+\b', answer.lower()))

            common_terms = key_terms.intersection(submission_terms)

            if len(common_terms) / max(1, len(key_terms)) > 0.7:  # 70% overlap
                quality_reward = 1.0  # Good match with true answer
            elif len(common_terms) / max(1, len(key_terms)) > 0.3:  # 30% overlap
                quality_reward = 0.7  # Partial match
            else:
                quality_reward = 0.3  # Poor match
        elif has_relevant_info:
            quality_reward = 0.8  # Did research but no true answer to compare

        observation = f"Answer submitted: {answer}"
        if true_answer:
            observation += f"\nReference answer: {true_answer}"
        if has_relevant_info:
            observation += "\nAnswer based on collected information."

        observation += "\nTask completed."

        return observation, quality_reward, True

    def observation(self, env_id: str):
        """
        Get environment observation

        Args:
            env_id: Environment ID

        Returns:
            str: Formatted observation
        """
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            env = self.env_instances[env_id]

            # Check if this is a new task
            if env["state"]["steps_taken"] == 0:
                return self._format_new_task(env_id)

            # Return current state observation
            return self._format_observation(env_id)

    def _format_new_task(self, env_id: str):
        """
        Format observation for a new task

        Args:
            env_id: Environment ID

        Returns:
            str: Formatted observation
        """
        env = self.env_instances[env_id]

        observation = (
            "New task starts.\n\n"
            f"Question: {env['question']}\n"
        )

        # Add context if available
        if env["state"]["context"]:
            observation += f"\nContext: {env['state']['context']}\n"

        # Add file information if available
        if env["state"]["files"]:
            observation += f"\nFiles available: {env['state']['files']}\n"

        # Add available tools information
        tools_str = ", ".join(env["available_tools"])
        observation += f"\nAvailable tools: {tools_str}\n"

        # Add tool descriptions
        observation += "\nTool descriptions:\n"
        for tool_name in env["available_tools"]:
            description = self.tool_manager.get_tool_description(tool_name)
            observation += f"- {tool_name}: {description}\n"

        observation += "\nUse tools in the format: Action: tool_name with Action Input: your_input\n"
        observation += "\nGive me one action."

        return observation

    def _format_observation(self, env_id: str):
        """
        Format regular observation

        Args:
            env_id: Environment ID

        Returns:
            str: Formatted observation
        """
        env = self.env_instances[env_id]

        observation = (
            f"Current task: {env['question']}\n"
            f"Steps taken: {env['state']['steps_taken']}\n"
        )

        # Add recent memory items (last 3 steps)
        if env["state"]["memory"]:
            observation += "\nRecent actions:\n"
            for memory_item in env["state"]["memory"][-3:]:
                observation += f"- Step {memory_item['step']}: {memory_item['action']}\n"
                observation += f"  Result: {memory_item['result']}\n"

        # Add tools reminder
        observation += f"\nAvailable tools: {', '.join(env['available_tools'])}\n"
        observation += "\nGive me one action."

        return observation

    def get_available_actions(self, env_id: str):
        """
        Get available actions for the environment

        Args:
            env_id: Environment ID

        Returns:
            list: Available actions
        """
        self._check_env_id(env_id)

        with self.env_locks[env_id]:
            return self.env_instances[env_id]["available_tools"]

    def _check_env_id(self, env_id: str):
        """
        Check if environment ID exists

        Args:
            env_id: Environment ID

        Raises:
            ValueError: If environment doesn't exist
        """
        if env_id not in self.env_instances:
            raise ValueError(f"Environment with ID {env_id} does not exist")
