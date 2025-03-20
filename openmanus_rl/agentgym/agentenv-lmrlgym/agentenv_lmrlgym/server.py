from fastapi import FastAPI
from .maze.environment import maze_server
from .maze.model import *
from .wordle.environment import wordle_server
from .wordle.model import *

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment LMRL-Gym."


# ----------------------------------------
# maze
# ----------------------------------------


@app.post("/maze/create")
async def maze_create():
    return maze_server.create()


@app.post("/maze/step")
def maze_step(body: MazeStepRequestBody):
    return maze_server.step(body.id, body.action)


@app.post("/maze/reset")
def maze_reset(body: MazeResetRequestBody):
    return maze_server.reset(body.id, body.game)


@app.get("/maze/available_actions")
def maze_get_available_actions():
    return maze_server.get_available_actions()


@app.get("/maze/observation")
def maze_get_observation(id: int):
    return maze_server.get_observation(id)


@app.get("/maze/detail")
def maze_get_detailed_info(id: int):
    return maze_server.get_detailed_info(id)


# ----------------------------------------
# wordle
# ----------------------------------------


@app.post("/wordle/create")
async def wordle_create():
    return wordle_server.create()


@app.post("/wordle/step")
def wordle_step(body: WordleStepRequestBody):
    return wordle_server.step(body.id, body.action)


@app.post("/wordle/reset")
def wordle_reset(body: WordleResetRequestBody):
    return wordle_server.reset(body.id, body.seed)


@app.get("/wordle/filtered_vocab")
def wordle_get_filtered_vocab(id: int):
    return wordle_server.get_filtered_vocab(id)


@app.get("/wordle/observation")
def wordle_get_observation(id: int):
    return wordle_server.get_observation(id)


@app.get("/wordle/detail")
def wordle_get_detailed_info(id: int):
    return wordle_server.get_detailed_info(id)
