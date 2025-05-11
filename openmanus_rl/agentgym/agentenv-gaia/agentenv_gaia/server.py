import asyncio
import json
import os
import re
import threading
import uuid
import uvicorn

from agentenv_gaia.environment import GaiaEnvServer
from agentenv_gaia.model import CreateQuery, StepQuery, StepResponse, ResetQuery


from fastapi import FastAPI, HTTPException

# Create FastAPI application
app = FastAPI(title="GAIA Environment Server")


# Create environment server instance
gaia_env_server = GaiaEnvServer()


# API Endpoints
@app.get("/")
def generate_ok():
    """Test connection"""
    return "ok"


@app.get("/list_envs")
def list_envs():
    """List all environments"""
    return list(gaia_env_server.env_instances.keys())


@app.post("/create")
def create(create_query: CreateQuery):
    """Create new environment"""
    try:
        env_id = gaia_env_server.create(
            create_query.id, create_query.dataset_type, create_query.tool_list
        )
        return env_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(step_query: StepQuery):
    """Execute environment step"""
    try:
        observation, reward, done, info = gaia_env_server.step(
            step_query.env_idx, step_query.action
        )
        return StepResponse(
            observation=observation, reward=reward, done=done, info=info
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/observation")
def observation(env_idx: str):
    """Get environment observation"""
    try:
        return gaia_env_server.observation(env_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(reset_query: ResetQuery):
    """Reset environment"""
    try:
        gaia_env_server.reset(
            reset_query.env_idx, reset_query.id, reset_query.dataset_type
        )
        return gaia_env_server.observation(reset_query.env_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/available_actions")
def available_actions(env_idx: str):
    """Get available actions for an environment"""
    try:
        return gaia_env_server.get_available_actions(env_idx)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Server launch function
def launch(host: str = "0.0.0.0", port: int = 8000):
    """Launch the GAIA environment server"""
    uvicorn.run(
        "agentenv_gaia.server:app",
        host=host,
        port=port,
        reload=False,
    )
