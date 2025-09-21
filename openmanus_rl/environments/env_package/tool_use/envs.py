"""
Tool Use Environment for complex reasoning tasks with tool calling capability.
Provides tasks from dataset and handles tool execution results.
"""

import json
import random
import importlib
from typing import List, Dict, Any, Tuple
import numpy as np


class ToolUseEnv:
    """
    Simple mock environment for tool use tasks.
    Provides tasks from dataset and handles tool execution.
    """
    
    def __init__(self, tasks_data: List[Dict], available_tools: List[str], seed: int = 42, model_string: str = None):
        self.tasks_data = tasks_data
        self.available_tools = available_tools
        self.tool_manager = ToolManager(available_tools, model_string=model_string)
        self.current_task_idx = 0
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def reset(self, task_idx: int = None) -> Tuple[str, Dict]:
        """Reset environment with a new task"""
        if task_idx is not None:
            self.current_task_idx = task_idx
        elif self.current_task_idx >= len(self.tasks_data):
            self.current_task_idx = 0
        
        task_data = self.tasks_data[self.current_task_idx]
        self.current_task_idx += 1
        
        # Return empty observation (task info is in info dict)
        info = {
            'task': task_data['question'],
            'answer': task_data['answer'], 
            'pid': task_data['pid'],
            'available_tools': self.available_tools,
            'tool_metadata': self.tool_manager.get_tools_metadata()
        }
        
        return "", info
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute action and return observation, reward, done, info.
        For tool use environment, we don't actually step - the environment manager handles everything.
        """
        return "", 0.0, False, {}
    
    def close(self):
        """Close environment"""
        pass


class ToolUseEnvs:
    """
    Vectorized wrapper for tool use environments.
    Similar to AlfworldEnvs but simpler since we don't need Ray workers.
    """
    
    def __init__(self, tasks_data: List[Dict], available_tools: List[str], 
                 seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, model_string: str = None):
        self.tasks_data = tasks_data
        self.available_tools = available_tools
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        
        # Create individual environments
        self.envs = []
        for i in range(self.num_processes):
            env = ToolUseEnv(tasks_data, available_tools, seed + i, model_string=model_string)
            self.envs.append(env)
        
        # Track current task indices for each environment
        self.current_indices = list(range(self.num_processes))
    
    def reset(self) -> Tuple[List[str], List[Dict]]:
        """Reset all environments"""
        obs_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            # Use different tasks for different environments
            task_idx = (self.current_indices[i]) % len(self.tasks_data)
            obs, info = env.reset(task_idx=task_idx)
            obs_list.append(obs)
            info_list.append(info)
            self.current_indices[i] += 1
        
        return obs_list, info_list
    
    def step(self, actions: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """Step all environments - placeholder since real logic is in environment manager"""
        obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            info_list.append(info)
        
        return obs_list, rewards_list, dones_list, info_list
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


class ToolManager:
    """Manages available tools and their execution"""
    
    def __init__(self, tool_names: List[str], model_string: str = None):
        self.tool_names = tool_names
        self.model_string = model_string
        self.available_tools = {}
        self._load_tools()
    
    def _load_tools(self):
        """Load tools specified in tool_names"""
        for tool_name in self.tool_names:
            try:
                self._load_tool(tool_name)
            except Exception as e:
                print(f"Warning: Failed to load tool '{tool_name}': {e}")
    
    def _load_tool(self, tool_name: str):
        """Load a specific tool"""
        # Map tool names to their module paths
        tool_mapping = {
            'google_search': 'openmanus_rl.tools.google_search.tool.Google_Search_Tool',
            'wikipedia_knowledge_searcher': 'openmanus_rl.tools.wikipedia_knowledge_searcher.tool.Wikipedia_Knowledge_Searcher_Tool',
            'arxiv_paper_searcher': 'openmanus_rl.tools.arxiv_paper_searcher.tool.ArXiv_Paper_Searcher_Tool',
            'pubmed_search': 'openmanus_rl.tools.pubmed_search.tool.Pubmed_Search_Tool',
            'url_text_extractor': 'openmanus_rl.tools.url_text_extractor.tool.URL_Text_Extractor_Tool',
            'python_code_generator': 'openmanus_rl.tools.python_code_generator.tool.Python_Code_Generator_Tool',
        }
        
        if tool_name not in tool_mapping:
            print(f"Unknown tool: {tool_name}, skipping...")
            return
        
        print(f"Loading tool: {tool_name}")
        
        module_path = tool_mapping[tool_name]
        module_name, class_name = module_path.rsplit('.', 1)
        
        # Import and instantiate the tool
        module = importlib.import_module(module_name)
        tool_class = getattr(module, class_name)
        
        # Check if tool requires LLM engine and pass model_string if available
        if hasattr(tool_class, 'require_llm_engine') and tool_class.require_llm_engine and self.model_string:
            tool_instance = tool_class(model_string=self.model_string)
        else:
            tool_instance = tool_class()
        
        self.available_tools[tool_name] = tool_instance
    
    def get_tools_metadata(self) -> str:
        """Generate formatted metadata for all available tools"""
        if not self.available_tools:
            return "No tools available."
        
        metadata_lines = []
        for tool_name, tool_instance in self.available_tools.items():
            metadata = tool_instance.get_metadata()
            
            tool_info = f"""Tool: {tool_name}
Description: {metadata.get('tool_description', 'No description')}
Input Types: {metadata.get('input_types', 'No input types specified')}
Usage: <tool_call>
tool: {tool_name}
parameters: {{"param_name": "param_value"}}
</tool_call>"""
            
            metadata_lines.append(tool_info.strip())
        
        return "\n\n".join(metadata_lines)
    
    def execute_tool(self, tool_name: str, params: Dict) -> str:
        """Execute a tool with given parameters"""
        if tool_name not in self.available_tools:
            return f"Error: Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
        
        try:
            tool_instance = self.available_tools[tool_name]
            result = tool_instance.execute(**params)
            
            # Convert result to string if needed
            if isinstance(result, (list, dict)):
                return json.dumps(result, indent=2, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


def build_tool_use_envs(tasks_data: List[Dict], available_tools: List[str], 
                       seed: int, env_num: int, group_n: int, is_train: bool = True, model_string: str = None):
    """Build tool use environments"""
    return ToolUseEnvs(tasks_data, available_tools, seed, env_num, group_n, is_train, model_string=model_string)
