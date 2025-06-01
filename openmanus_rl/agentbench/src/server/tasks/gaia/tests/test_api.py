#!/usr/bin/env python
"""
Comprehensive manual test script for GAIA Environment Server

This script tests all functionality of the GAIA Environment Server including:
- Environment creation, reset, and observation
- All available tools (web_search, bash, python_execute, terminate)
- Different action formats (standard, JSON, direct format)
- Error handling
"""

import requests
import json
import sys
import time
import datetime
from pprint import pformat

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
VERBOSE_LOGGING = True  # Set to True for full request/response logging
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

def get_observation(env_id):
    """Get observation for environment"""
    print_step(f"Getting observation for environment {env_id}...")
    
    observation = make_request("GET", "observation", params={"env_idx": env_id})
    
    if observation:
        print_success("Observation received")
        print(f"\n{Colors.CYAN}{'='*40} OBSERVATION {'='*40}{Colors.ENDC}")
        print(f"{Colors.CYAN}{observation}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*90}{Colors.ENDC}\n")
        return observation
    else:
        print_error("Failed to get observation")
        return None

def get_available_actions(env_id):
    """Get available actions for environment"""
    print_step(f"Getting available actions for environment {env_id}...")
    
    actions = make_request("GET", "available_actions", params={"env_idx": env_id})
    
    if actions:
        print_success(f"Available actions: {', '.join(actions)}")
        return actions
    else:
        print_error("Failed to get available actions")
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

def reset_environment(env_id, task_id=None, dataset_type="validation"):
    """Reset environment to a new or same task"""
    if task_id is not None:
        print_step(f"Resetting environment {env_id} to new task (id={task_id}, dataset={dataset_type})...")
    else:
        print_step(f"Resetting environment {env_id} to same task...")
    
    data = {"env_idx": env_id}
    if task_id is not None:
        data["id"] = task_id
        data["dataset_type"] = dataset_type
    
    observation = make_request("POST", "reset", data=data)
    
    if observation:
        print_success("Environment reset successfully")
        print(f"\n{Colors.CYAN}{'='*40} NEW OBSERVATION {'='*40}{Colors.ENDC}")
        print(f"{Colors.CYAN}{observation}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*90}{Colors.ENDC}\n")
        return observation
    else:
        print_error("Failed to reset environment")
        return None

def run_tests():
    """Run all tests"""
    # Store test results
    test_results = {}
    start_time = time.time()
    
    # Print test header
    print_section("GAIA Environment Server API Tests")
    print(f"{get_timestamp()}Starting tests at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{get_timestamp()}Server URL: {BASE_URL}")
    
    # Test server connection
    print_section("Server Connection Test")
    if not test_server_connection():
        print_error("Aborting tests due to connection failure")
        return
    test_results["server_connection"] = True
    
    # Test environment creation
    print_section("Environment Creation Test")
    env_id = create_environment()
    if not env_id:
        print_error("Aborting tests due to environment creation failure")
        return
    test_results["env_creation"] = True
    
    # Test custom tools creation
    print_section("Custom Tools Environment Test")
    custom_tools = ["web_search", "python_execute", "terminate"]
    custom_tools_env_id = create_environment(
        task_id=1, 
        tool_list=custom_tools
    )
    if custom_tools_env_id:
        print_success(f"Successfully created environment with custom tools: {', '.join(custom_tools)}")
        test_results["custom_tools_creation"] = True
    else:
        print_error("Failed to create environment with custom tools")
        test_results["custom_tools_creation"] = False
    
    # Get initial observation and available actions
    print_section("Initial Observation Test")
    observation = get_observation(env_id)
    test_results["get_observation"] = observation is not None
    
    actions = get_available_actions(env_id)
    test_results["get_available_actions"] = actions is not None
    
    # Test all tools with different formats
    
    # 1. Test web_search with standard format
    # 1.1 Tool argument is fault
    print_section("Web Search Tool Test (standard format, faulty tool argument)")
    web_search_action = "Action: web_search with Action Input: What is the capital of France?"
    web_search_result = execute_action(
        env_id, 
        web_search_action, 
        format_type="standard",
        description="web search for 'What is the capital of France?'"
    )
    test_results["web_search_tool"] = web_search_result is not None

    # 1.2 Tool argument is correct
    print_section("Web Search Tool Test (standard format, correct tool argument)")
    web_search_action = "Action: web_search Action Input: What is the capital of France?"
    web_search_result = execute_action(
        env_id, 
        web_search_action, 
        format_type="standard",
        description="web search for 'What is the capital of France?'"
    )
    test_results["web_search_tool"] = web_search_result is not None
    
    # 2. Test bash with JSON format
    print_section("Bash Tool Test (JSON Format)")
    bash_action = {
        "tool_name": "bash",
        "command": "echo 'Hello from bash test' && ls -la | head -n 5"
    }
    bash_result = execute_action(
        env_id, 
        bash_action, 
        format_type="json",
        description="bash command to echo text and list files"
    )
    test_results["bash_tool"] = bash_result is not None
    
    # 3. Test python_execute with direct format
    print_section("Python Execute Tool Test (Direct Format)")
    python_action = {
        "tool_name": "python_execute",
        "code": "import random\nprint('Random number:', random.randint(1, 100))\nprint('Test successful!')"
    }
    python_result = execute_action(
        env_id, 
        python_action, 
        format_type="direct",
        description="Python code to generate a random number"
    )
    test_results["python_execute_tool"] = python_result is not None
    
    # 3.1 Test browser_use with JSON format, go to url
    print_section("Browser Use Tool Test (JSON Format, go to url)")
    browser_action = {
        "tool_name": "browser_use",
        "action": "go_to_url",
        "url": "https://www.example.com"
    }
    browser_result = execute_action(
        env_id, 
        browser_action, 
        format_type="json",
        description="browser action to navigate to example.com"
    )
    test_results["browser_use_tool_go_to_url"] = browser_result is not None
    
    # 3.2 Test browser_use with JSON format, web search
    print_section("Browser Use Tool Test (JSON Format, Web Search)")
    browser_json_action = {
        "tool_name": "browser_use",
        "action": "web_search",
        "query": "How to test browser automation"
    }
    browser_json_result = execute_action(
        env_id, 
        browser_json_action, 
        format_type="json",
        description="browser action to search the web"
    )
    test_results["browser_use_tool_web_search"] = browser_json_result is not None

    # 3.3 Test browser_use with JSON format, extract content
    print_section("Browser Use Tool Test (JSON Format, Extract Content)")
    browser_json_action = {
        "tool_name": "browser_use",
        "action": "extract_content",
        "goal": "extract the content of the page"
    }
    browser_json_result = execute_action(
        env_id, 
        browser_json_action, 
        format_type="json",
        description="browser action to extract content from example.com"
    )
    test_results["browser_use_tool_extract_content"] = browser_json_result is not None

    # 4. Test environment reset
    print_section("Environment Reset Test")
    reset_result = reset_environment(env_id, task_id=1)
    test_results["env_reset"] = reset_result is not None
    
    # 5. Test terminate tool
    print_section("Terminate Tool Test (json format, correct tool argument)")
    terminate_action = {
        "tool_name": "terminate",
        "status": "success"
    }
    terminate_result = execute_action(
        env_id, 
        terminate_action,
        format_type="json",
        description="terminate action with final answer"
    )
    test_results["terminate_tool"] = terminate_result is not None
    test_results["terminate_done_state"] = terminate_result is not None and terminate_result.get("done", False)
    
    # 6. Test error handling for invalid tool
    print_section("Invalid Tool Test")
    invalid_action = "Action: invalid_tool Action Input: This should fail gracefully"
    invalid_result = execute_action(
        env_id, 
        invalid_action,
        description="invalid tool that should be handled gracefully"
    )
    # This should return a result with an error message but not crash
    test_results["invalid_tool_handled"] = invalid_result is not None
    
    # 7. Test listing environments
    print_section("List Environments Test")
    print_step("Listing all active environments...")
    envs = make_request("GET", "list_envs")
    if envs:
        print_success(f"Environment listing successful: {envs}")
        # Verify both environments are in the list
        env_ids_found = [env_id in envs, custom_tools_env_id in envs]
        if all(env_ids_found):
            print_success("All created environments found in list")
        else:
            print_warning("Not all created environments were found in the list")
        test_results["list_envs"] = True
    else:
        print_error("Failed to list environments")
        test_results["list_envs"] = False
    
    # Calculate test duration
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Summary
    print_section("Test Summary")
    print(f"{get_timestamp()}Tests completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{get_timestamp()}Total test duration: {test_duration:.2f} seconds")
    print(f"{get_timestamp()}Created environments: {env_id}, {custom_tools_env_id}")
    
    # Count successes and failures
    success_count = sum(1 for result in test_results.values() if result is True)
    failure_count = sum(1 for result in test_results.values() if result is False)
    
    print(f"\n{Colors.BOLD}Test Results:{Colors.ENDC}")
    print(f"  {Colors.GREEN}Passed: {success_count}/{len(test_results)}{Colors.ENDC}")
    print(f"  {Colors.RED if failure_count > 0 else ''}Failed: {failure_count}/{len(test_results)}{Colors.ENDC}")
    
    # Print detailed test results
    print(f"\n{Colors.BOLD}Detailed Test Results:{Colors.ENDC}")
    for test_name, result in test_results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if result else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
        print(f"  {test_name}: {status}")
    
    if test_results.get("terminate_done_state", False):
        print_success("\nEnvironment successfully completed task with terminate action")
    else:
        print_warning("\nEnvironment did not reach completion state with terminate action")

if __name__ == "__main__":
    try:
        # Check if running in CI mode (without colors)
        if "--no-color" in sys.argv:
            # Disable colors
            for attr in dir(Colors):
                if not attr.startswith('__'):
                    setattr(Colors, attr, '')
        
        # Check for verbose mode
        if "--verbose" in sys.argv:
            VERBOSE_LOGGING = True
            LOG_REQUEST_BODY = True
            LOG_RESPONSE_BODY = True
        elif "--quiet" in sys.argv:
            VERBOSE_LOGGING = False
            LOG_REQUEST_BODY = False
            LOG_RESPONSE_BODY = False
        
        run_tests()
    except KeyboardInterrupt:
        print_warning("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error during tests: {str(e)}")
        sys.exit(1)
