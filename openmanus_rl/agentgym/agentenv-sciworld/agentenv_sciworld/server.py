from fastapi import FastAPI
from .model import *
from .environment import server

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment ScienceWorld."


@app.post("/create")
async def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def get_observation(id: int):
    return server.get_observation(id)


@app.get("/action_hint")
def get_action_hint(id: int):
    return server.get_action_hint(id)


@app.get("/goals")
def get_goals(id: int):
    return server.get_goals(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)
