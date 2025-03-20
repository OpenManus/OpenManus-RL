import re
import math
import os
from utils.logging.agent_logger import AgentLogger

def extract_action_name_and_action_input(text):
    pattern = r"\s*(.*?)\s*with\s*Action Input:\s*(.*?)$"
    match = re.search(pattern, text)
    if match:
        action = match.group(1)
        action_input = match.group(2)
        return action, action_input
    else:
        return None, None

def parse_action(string):
    pattern = r".*?Action:\s*(.*?)\s*with\s*Action Input:\s*(.*?)$"
    match = re.match(pattern, string, re.MULTILINE | re.DOTALL)
    if match:
        action_type = match.group(1)
        params = match.group(2)
        try:
            params = eval(params)
            # convert all value to string
            # for key, value in params.items():
                # params[key] = str(value)
        except:
            raise Exception("Parameters in action input are not valid, please change your action input.")
        return action_type, params

    else:
        return None

# extract current sheet id and open sheet first
def extract_sheet_number(s):
    match = re.search(r'"Sheet(\d{1,2})"', s)
    if match:
        return "Sheet" + match.group(1)
    else:
        return None

def is_same_location(coord1, coord2, threshold=50):
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    radius = 6371

    distance = radius * c

    return distance < threshold


def check_credentials():
    if "MOVIE_KEY" not in os.environ:
        raise Exception("Please set MOVIE_KEY in `.env` .")
    
    if "TODO_KEY" not in os.environ:
        raise Exception("Please set TODO_KEY in `.env` .")
    
    PROJECT_PATH = os.getenv("PROJECT_PATH")
    if not os.path.exists(f"{PROJECT_PATH}/toolusage/utils/sheet/credential.json"):
        raise Exception("Please set `credential.json` in `./toolusage/utils/sheet/credential.json` .")

def contains_network_error(observation):
    network_errors = [
        "ConnectionError",
        "HTTPError",
        "HTTPSError",
        "TimeoutError",
        "SSLError",
        "ProxyError",
        "TooManyRedirects",
        "RequestException"
    ]

    for error in network_errors:
        if error in observation:
            return True

    return False

def save_log(logger_name, task_name, output_dir):
    """Creates a log file and logging object for the corresponding task Name"""
    log_dir = os.path.join(output_dir, 'trajectory')
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f'{task_name}.log'
    log_file_path = os.path.join(log_dir, log_file_name)
    logger = AgentLogger(logger_name, filepath=log_file_path)
    return logger 