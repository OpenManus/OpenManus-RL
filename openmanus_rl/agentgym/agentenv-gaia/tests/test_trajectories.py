#!/usr/bin/env python
"""
Test script for executing example trajectory files against the GAIA Environment Server

This script tests the GAIA Environment Server by replaying trajectory examples from:
- bash_traj.json
- python_execute_traj.json
- web_search_traj.json
- browser_use_traj.json

It makes HTTP requests to a running server instance and checks if the responses
match the expected behavior for each tool.
"""

import os
import json
import sys
import time
import datetime
import argparse
from pprint import pformat
import requests

# Server URL
BASE_URL = "http://localhost:8000"

# Define colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'

# Configure logging details
VERBOSE_LOGGING = True
LOG_REQUEST_BODY = True
LOG_RESPONSE_BODY = True
LOG_TIMESTAMPS = True

def get_timestamp():
    """Get formatted timestamp"""
    if LOG_TIMESTAMPS:
        return f"{Colors.GRAY}[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}]{Colors.ENDC} "
    return ""

def print_section(title):
    """Print a section header"""
    print(f"\n{get_timestamp()}{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{get_timestamp()}{Colors.HEADER}{Colors.BOLD} {title} {Colors.ENDC}")
    print(f"{get_timestamp()}{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_step(step):
    """Print a test step"""
    print(f"{get_timestamp()}{Colors.BLUE}{Colors.BOLD}>>> {step}{Colors.ENDC}")

def print_success(message):
    """Print a success message"""
    print(f"{get_timestamp()}{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{get_timestamp()}{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{get_timestamp()}{Colors.RED}✖ {message}{Colors.ENDC}")

def print_debug(message):
    """Print a debug message"""
    if VERBOSE_LOGGING:
        print(f"{get_timestamp()}{Colors.GRAY}DEBUG: {message}{Colors.ENDC}")

def format_json(obj):
    """Format JSON for pretty printing"""
    if isinstance(obj, (dict, list)):
        return pformat(obj, indent=2)
    return str(obj)

def make_request(method, endpoint, data=None, params=None):
    """Make a request to the server with error handling"""
    url = f"{BASE_URL}/{endpoint}"
    
    # Print request details
    print_debug(f"Request: {method} {url}")
    if params and VERBOSE_LOGGING:
        print_debug(f"Query params: {format_json(params)}")
    if data and LOG_REQUEST_BODY:
        print_debug(f"Request body: {format_json(data)}")
    
    start_time = time.time()
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print_error(f"Unknown method: {method}")
            return None
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        print_debug(f"Response status: {response.status_code} (in {response_time:.2f}ms)")
        
        response.raise_for_status()
        
        # Print response body if enabled
        if LOG_RESPONSE_BODY:
            try:
                response_json = response.json()
                print_debug(f"Response body: {format_json(response_json)}")
                return response_json
            except json.JSONDecodeError:
                print_debug(f"Response (not JSON): {response.text}")
                return response.text
        else:
            return response.json()
    except requests.exceptions.ConnectionError:
        print_error(f"Connection refused. Is the server running at {BASE_URL}?")
        return None
    except requests.exceptions.HTTPError as e:
        print_error(f"HTTP Error: {e}")
        print_debug(f"Response body: {response.text}")
        return None
    except json.JSONDecodeError:
        print_error(f"Invalid JSON response: {response.text}")
        return None
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return None

def test_server_connection():
    """Test basic server connection"""
    print_step("Testing server connection...")
    response = make_request("GET", "")
    
    if response == "ok":
        print_success("Server connection successful!")
        return True
    else:
        print_error("Server connection failed!")
        return False

def create_environment(task_id=0, dataset_type="validation", tool_list=None):
    """Create a test environment"""
    print_step(f"Creating environment (task_id={task_id}, dataset={dataset_type}, tools={tool_list if tool_list else 'default'})...")
    
    data = {"id": task_id, "dataset_type": dataset_type}
    if tool_list:
        data["tool_list"] = tool_list
    
    env_id = make_request("POST", "create", data=data)
    
    if env_id:
        print_success(f"Environment created with ID: {env_id}")
        return env_id
    else:
        print_error("Failed to create environment")
        return None

def execute_action(env_id, action, format_type="standard", description=None):
    """Execute an action in the environment with different formats"""
    if description:
        print_step(f"Executing {description} ({format_type} format)...")
    else:
        print_step(f"Executing action ({format_type} format)...")
    
    if format_type == "standard":
        # Standard format: "Action: tool_name with Action Input: input"
        formatted_action = action
    elif format_type == "json":
        # JSON format with tool_name and parameters
        formatted_action = json.dumps(action)
    elif format_type == "direct":
        # Direct format: "tool_name: input"
        tool_name = action.get("tool_name", "")
        if tool_name == "web_search":
            formatted_action = f"web_search: {action.get('query', '')}"
        elif tool_name == "bash":
            formatted_action = f"bash: {action.get('command', '')}"
        elif tool_name == "python_execute":
            formatted_action = f"python_execute: {action.get('code', '')}"
        elif tool_name == "browser_use":
            # For browser_use, convert args to JSON string
            args = {k: v for k, v in action.items() if k != "tool_name"}
            formatted_action = f"browser_use: {json.dumps(args)}"
        elif tool_name == "terminate":
            formatted_action = f"terminate: {action.get('answer', '')}"
    
    print(f"{get_timestamp()}{Colors.CYAN}Action: {formatted_action}{Colors.ENDC}")
    
    data = {"env_idx": env_id, "action": formatted_action}
    result = make_request("POST", "step", data=data)
    
    if result:
        print_success("Action executed successfully")
        
        print(f"\n{Colors.CYAN}{'='*40} RESULT {'='*40}{Colors.ENDC}")
        print(f"{Colors.CYAN}Observation: {result['observation']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*90}{Colors.ENDC}\n")
        
        # Print detailed result information
        print(f"Reward: {result['reward']}", end="")
        if result['reward'] > 0:
            print(f" {Colors.GREEN}(positive){Colors.ENDC}")
        elif result['reward'] < 0:
            print(f" {Colors.RED}(negative){Colors.ENDC}")
        else:
            print(f" {Colors.YELLOW}(neutral){Colors.ENDC}")
            
        print(f"Done: {result['done']}")
        
        # Print step information if available
        if 'info' in result and 'steps_taken' in result['info']:
            print(f"Steps taken: {result['info']['steps_taken']}")
        
        return result
    else:
        print_error("Failed to execute action")
        return None

class TrajectoryTester:
    """Class for executing and validating example trajectories"""
    
    def __init__(self):
        """Initialize the trajectory tester"""
        self.trajectories = {
            "bash": self._load_trajectory("bash_traj.json"),
            "python_execute": self._load_trajectory("python_execute_traj.json"),
            "web_search": self._load_trajectory("web_search_traj.json"),
            "browser_use": self._load_trajectory("browser_use_traj.json")
        }
        self.results = {}
        
    def _load_trajectory(self, filename: str) -> dict:
        """Load trajectory from JSON file"""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        traj_path = os.path.join(test_dir, filename)
        
        try:
            with open(traj_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print_warning(f"Failed to load trajectory file {filename}: {str(e)}")
            return {"messages": []}
    
    def _extract_tool_calls(self, trajectory: dict) -> list:
        """Extract tool calls from a trajectory"""
        tool_calls = []
        for message in trajectory.get("messages", []):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls", []):
                    if tool_call.get("function", {}).get("name") and tool_call.get("function", {}).get("arguments"):
                        tool_calls.append({
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"])
                        })
        return tool_calls
    
    def _format_action(self, tool_call: dict) -> dict:
        """Format a tool call into an action dictionary for execute_action"""
        tool_name = tool_call["name"]
        args = tool_call["arguments"]
        
        # Format all tool calls as JSON
        action = {"tool_name": tool_name}
        
        if tool_name == "bash":
            action["command"] = args.get("command", "")
        elif tool_name == "python_execute":
            action["code"] = args.get("code", "")
        elif tool_name == "web_search":
            action["query"] = args.get("query", "")
        elif tool_name == "browser_use":
            # For browser_use, include all arguments
            action.update(args)
        elif tool_name == "terminate":
            action["status"] = args.get("status", "success")
            
        return action
    
    def test_trajectory(self, traj_name: str, env_id: str = None) -> bool:
        """Test a specific trajectory"""
        print_section(f"Testing {traj_name.upper()} Trajectory")
        
        # Get the trajectory
        trajectory = self.trajectories.get(traj_name)
        if not trajectory or not trajectory.get("messages"):
            print_error(f"No valid trajectory found for {traj_name}")
            self.results[traj_name] = False
            return False
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(trajectory)
        if not tool_calls:
            print_error(f"No tool calls found in {traj_name} trajectory")
            self.results[traj_name] = False
            return False
            
        print_success(f"Found {len(tool_calls)} tool calls in trajectory")
        
        # Get required tool list for this trajectory
        tool_names = set(tc["name"] for tc in tool_calls)
        tool_list = list(tool_names)
        print_debug(f"Required tools for trajectory: {', '.join(tool_list)}")
        
        # Create environment if not provided
        if env_id is None:
            env_id = create_environment(tool_list=tool_list)
            if not env_id:
                print_error(f"Failed to create environment for {traj_name} trajectory")
                self.results[traj_name] = False
                return False
        
        # Process each tool call
        success = True
        for i, tool_call in enumerate(tool_calls):
            action = self._format_action(tool_call)
            
            # Create a description for the action
            tool_type = tool_call["name"]
            description = f"{tool_type} action (step {i+1}/{len(tool_calls)})"
            
            # Execute the action
            result = execute_action(env_id, action, format_type="json", description=description)
            
            if not result:
                print_error(f"Failed to execute {tool_type} action")
                success = False
                break
                
            # Check for trajectory completion
            if i == len(tool_calls) - 1 and tool_type == "terminate":
                if not result.get("done", False):
                    print_warning("Trajectory did not reach 'done' state with terminate action")
                    success = False
                else:
                    print_success("Trajectory successfully completed with terminate action")
        
        # Record result
        self.results[traj_name] = success
        return success
    
    def test_all_trajectories(self):
        """Test all available trajectories"""
        all_success = True
        
        # Test each trajectory type
        for traj_name in self.trajectories.keys():
            success = self.test_trajectory(traj_name)
            all_success = all_success and success
        
        # Print summary
        print_section("Trajectory Tests Summary")
        print(f"{Colors.BOLD}Results:{Colors.ENDC}")
        for traj_name, result in self.results.items():
            status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
            print(f"  {traj_name.ljust(15)}: {status}")
        
        return all_success

def run_tests():
    """Run all trajectory tests"""
    start_time = time.time()
    
    # Print test header
    print_section("GAIA Environment Trajectory Tests")
    print(f"{get_timestamp()}Starting tests at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{get_timestamp()}Server URL: {BASE_URL}")
    
    # Test server connection
    if not test_server_connection():
        print_error("Aborting tests due to connection failure")
        return False
    
    # Run trajectory tests
    tester = TrajectoryTester()
    success = tester.test_all_trajectories()
    
    # Calculate test duration
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Print test summary
    print(f"\n{get_timestamp()}Tests completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{get_timestamp()}Total test duration: {test_duration:.2f} seconds")
    
    return success

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="GAIA Environment Trajectory Tests")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Disable all but essential output")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the GAIA server")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Configure logging options
    if args.verbose:
        VERBOSE_LOGGING = True
        LOG_REQUEST_BODY = True
        LOG_RESPONSE_BODY = True
    elif args.quiet:
        VERBOSE_LOGGING = False
        LOG_REQUEST_BODY = False
        LOG_RESPONSE_BODY = False
    
    # Configure color options
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, '')
    
    # Set server URL
    BASE_URL = args.url
    
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error during tests: {str(e)}")
        sys.exit(1) 