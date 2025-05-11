# GAIA Environment Server Tests

This directory contains comprehensive test suites for the GAIA Environment Server implementation.

## Test Files

- `test_server.py`: Main test file containing unit tests for:
  - `ToolManager` class
  - `GaiaEnvServer` class
  - API endpoints
  - Trajectory examples

- `test_api.py`: Comprehensive API testing script with detailed logging

- `test_trajectories.py`: Tests server behavior with example trajectory files

- Example trajectory files:
  - `bash_traj.json`: Example of bash tool usage to print "hello world"
  - `python_execute_traj.json`: Example of Python execution (bubble sort implementation)
  - `web_search_traj.json`: Example of web search functionality
  - `browser_use_traj.json`: Example of browser automation and interaction

- Test runners:
  - `run_tests.py`: Script to run all unit tests automatically

## Running Tests

### Automated Unit Tests

```bash
# Run all tests
python tests/run_tests.py

# Or run with executable permission
./tests/run_tests.py
```

### Using pytest

```bash
# From the agentenv-gaia directory
pytest tests/test_server.py -v
pytest tests/test_trajectories.py -v
```

### Using unittest

```bash
# From the agentenv-gaia directory
python -m unittest tests/test_server.py
```

## Manual API Testing

The `test_api.py` script provides comprehensive interactive testing of the GAIA server API:

1. First, start the server:
```bash
python -m agentenv_gaia.server
```

2. Then run the test script in a different terminal:
```bash
python tests/test_api.py

# Or run with executable permission
./tests/test_api.py
```

### Command Line Options

The `test_api.py` script accepts several command-line arguments:

```bash
# For more verbose output including request/response details
./tests/test_api.py --verbose

# For minimal output (only errors and results)
./tests/test_api.py --quiet

# Without color coding (useful for CI environments)
./tests/test_api.py --no-color
```

### Features of the API Test Script

The `test_api.py` script tests:

- Server connection
- Environment creation with default and custom tools
- All available tools (web_search, bash, python_execute, browser_use, terminate)
- Different action formats (standard, JSON, direct)
- Error handling with invalid tools/actions
- Environment reset functionality
- List environments endpoint

The script includes:

- Time-stamped logging for all operations
- Color-coded output for different types of messages
- Detailed request and response information
- Test duration tracking
- Comprehensive result summary
- HTTP error handling with detailed diagnostics

## Test Coverage

The tests cover the following functionality:

1. Tool management:
   - Tool initialization
   - Tool execution
   - Parameter mapping

2. Environment operations:
   - Environment creation
   - State management
   - Action parsing and execution
   - Observation formatting

3. API endpoints:
   - Environment creation
   - Observation retrieval
   - Action execution
   - Environment reset

4. Trajectory examples:
   - Testing with real-world examples from provided JSON files

5. Browser automation (BrowserUseTool):
   - Web navigation (go_to_url)
   - Content extraction (extract_content)
   - Element interaction (click_element, input_text)
   - Scrolling operations (scroll_down, scroll_up, scroll_to_text)
   - Tab management (open_tab, close_tab, switch_tab)
   - Keyboard input (send_keys)
   - Dropdown handling (get_dropdown_options, select_dropdown_option) 