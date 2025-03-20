"""
Flask Server
"""

from typing import List, Tuple, Any

from flask import Flask, jsonify, request
import subprocess

from .environment import webarena_env_server
from .utils import debug_flg


app = Flask(__name__)


@app.route("/", methods=["GET"])
def generate_ok():
    """Test connectivity"""
    return "ok"


@app.route("/list_envs", methods=["GET"])
def list_envs():
    """List all environments. In fact, only one env is allowed."""
    return jsonify(list(webarena_env_server.env.keys()))


@app.route("/create", methods=["POST"])
def create():
    """Create a new environment"""
    subprocess.run(
        ["python", "browser_env/auto_login.py"]
    )  # This will take a long time.
    env = webarena_env_server.create()
    return jsonify({"env_idx": env})


@app.route("/step", methods=["POST"])
def step():
    """
    Make an action

    step_query:
    {
        "env_idx": int,
        "action": str, # llm output
    }

    """
    step_query = request.json
    step_data = webarena_env_server.step(step_query["env_idx"], step_query["action"])
    step_response = {}
    step_response["observation"] = step_data[0]
    step_response["reward"] = step_data[1]
    step_response["terminated"] = step_data[2]
    step_response["truncated"] = step_data[3]
    step_response["info"] = step_data[4]
    return jsonify(step_response)


@app.route("/observation", methods=["GET"])
def get_observation():
    """
    current observation
    """
    env_idx = request.args.get("env_idx", type=int)
    obs = webarena_env_server.observation(env_idx)
    return jsonify(obs)


@app.route("/observation_metadata", methods=["GET"])
def get_obsmetadata():
    """
    current observation metadata
    """
    env_idx = request.args.get("env_idx", type=int)
    obs_meta = webarena_env_server.observation_metadata(env_idx)
    return jsonify(obs_meta["text"])


@app.route("/reset", methods=["POST"])
def reset():
    """
    reset the environment
    reset_query:
    {
        "env_idx": int,
        "seed": int, # please sent 0
        "idx" : int,
    }
    """
    reset_query = request.json
    reset_query["options"] = {
        "config_file": "./config_files/{}.json".format(reset_query["idx"])
    }
    obs, info = webarena_env_server.reset(
        reset_query["env_idx"], reset_query["seed"], reset_query["options"]
    )
    reset_response = {}
    reset_response["observation"] = obs["text"]
    return jsonify(reset_response)


@app.route("/close", methods=["POST"])
def close():
    close_query = request.json
    webarena_env_server.close(close_query["env_idx"])
    return "closed"
