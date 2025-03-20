from fastapi import FastAPI
from .model import *
from .env_wrapper import server

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment TextCraft."


@app.post("/create")
async def create(body: CreateRequestBody):
    return server.create(body.commands, body.goal)


@app.post("/step")
def step(body: StepRequestBody):
    print(f"/step {body.id} {body.action}")
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    print(f"/reset {body.id} {body.data_idx}")
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def get_observation(id: int):
    print(f"/observation {id}")
    return server.get_observation(id)


@app.get("/commands")
def get_commands(id: int):
    return server.get_commands(id)


@app.get("/goal")
def get_goal(id: int):
    return server.get_goal(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)
