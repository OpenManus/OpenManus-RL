"""
FastAPI Server
"""

import logging
import time
from typing import List, Tuple

from fastapi import FastAPI, Request

from .environment import webshop_env_server
from .model import *
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# 自定义中间件
@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - {response.status_code} - {process_time:.2f} seconds"
    )
    return response


@app.get("/", response_model=str)
async def generate_ok():
    """Test connectivity"""
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    """List all environments"""
    return list(webshop_env_server.env.keys())


@app.post("/create", response_model=int)
async def create():
    """Create a new environment"""
    env = webshop_env_server.create()

    return env


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    print("/step")
    print(step_query.env_idx)
    print(step_query.action)
    state, reward, done, info = webshop_env_server.step(
        step_query.env_idx, step_query.action
    )
    print(step_query.env_idx)
    print(state)
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/available_actions", response_model=AvailableActionsResponse)
def get_available_actions(env_idx: int):
    res = webshop_env_server.get_available_actions(env_idx)
    has_search_bar = res["has_search_bar"]
    clickables = res["clickables"]
    return AvailableActionsResponse(
        has_search_bar=has_search_bar, clickables=clickables
    )


@app.get("/instruction_text", response_model=str)
def get_instruction_text(env_idx: int):
    print("/instruction_text")
    print(env_idx)
    res = webshop_env_server.get_instruction_text(env_idx)
    print(res)
    return res


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    print("/observation")
    print(env_idx)
    res = webshop_env_server.observation(env_idx)
    return res


@app.get("/state", response_model=StateResponse)
def get_state(env_idx: int):
    print("/state")
    url, html, instruction_text = webshop_env_server.state(env_idx)
    print(env_idx)
    print(instruction_text)
    return StateResponse(url=url, html=html, instruction_text=instruction_text)


@app.post("/reset", response_model=Tuple[str, None])
def reset(reset_query: ResetQuery):
    print(reset_query)
    return webshop_env_server.reset(reset_query.env_idx, reset_query.session_id)
