from fastapi import FastAPI
from .model import *
from .environment import server

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment BabyAI."


@app.post("/create")
async def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)

