"""
FastAPI Server
"""

from typing import List, Tuple

from fastapi import FastAPI

from .sheet_environment import sheet_env_server
from .sheet_model import *
from .sheet_utils import debug_flg

app = FastAPI(debug=debug_flg)


@app.get("/", response_model=str)
def generate_ok():
    """Test connectivity"""
    return "ok"


@app.get("/list_envs", response_model=List[int])
def list_envs():
    """List all environments"""
    return list(sheet_env_server.env.keys())


@app.post("/create", response_model=int)
def create(create_query: CreateQuery):
    """Create a new environment"""
    env = sheet_env_server.create(create_query.id)
    return env


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    observation, reward, done, _ = sheet_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(observation=observation, reward=reward, done=done)


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return sheet_env_server.observation(env_idx)


@app.post("/reset", response_model=str)
def reset(reset_query: ResetQuery):
    sheet_env_server.reset(reset_query.env_idx, reset_query.id)
    return sheet_env_server.observation(reset_query.env_idx)
