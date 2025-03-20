"""
FastAPI Server
"""

from typing import List

from fastapi import FastAPI

from .movie_environment import movie_env_server
from .movie_model import *
from .movie_utils import debug_flg

app = FastAPI(debug=debug_flg)


@app.get("/", response_model=str)
def generate_ok():
    """Test connectivity"""
    return "ok"


@app.get("/list_envs", response_model=List[int])
def list_envs():
    """List all environments"""
    return list(movie_env_server.env.keys())


@app.post("/create", response_model=int)
def create(create_query: CreateQuery):
    """Create a new environment"""
    env = movie_env_server.create(create_query.id)
    return env


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    observation, reward, done, _ = movie_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(observation=observation, reward=reward, done=done)


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return movie_env_server.observation(env_idx)


@app.post("/reset", response_model=str)
def reset(reset_query: ResetQuery):
    movie_env_server.reset(reset_query.env_idx, reset_query.id)
    return movie_env_server.observation(reset_query.env_idx)
