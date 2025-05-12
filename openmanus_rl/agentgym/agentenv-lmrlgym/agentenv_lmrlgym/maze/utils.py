import numpy as np
import random
from typing import Tuple, List, Dict
from llm_rl_scripts.maze.env.env import MazeEnv, describe_objects, manhatten_actions
from llm_rl_scripts.maze.env.mazes import (
    t_maze,
    u_maze,
    double_t_maze,
    random_shape_maze,
    random_maze,
)


def describe_observation(
    maze: np.ndarray,
    cur_position: Tuple[int, int] = None,
    goal_position: Tuple[int, int] = None,
    init_position: Tuple[int, int] = None,
    display_position: bool = True,
    display_init_position: bool = True,
):
    assert len(maze.shape) == 2
    position_description = init_description = ""
    if display_position:
        assert cur_position is not None, "cur_position is not given"
        assert goal_position is not None, "goal_position is not given"
        cur_description = f"Your current position is at position {cur_position[0]}, {cur_position[1]}. "
        goal_description = (
            f"The goal is at position {goal_position[0]}, {goal_position[1]}. "
        )
        position_description = goal_description + cur_description
    if display_init_position:
        assert init_position is not None, "init_position is not given"
        init_description = f"Your init position is at position {init_position[0]}, {init_position[1]}. "
    delta_descriptions = {
        "to your right": (0, 1),
        "to your left": (0, -1),
        "above you": (-1, 0),
        "below you": (1, 0),
    }
    walls = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[cur_position[0] + dy, cur_position[1] + dx] == 1:
            walls.append(k)
    wall_description = describe_objects("wall", walls)
    return position_description + init_description + wall_description


def setup_maze_env(
    maze_name: str,
    width: int = 13,
    height: int = 10,
    display_position: bool = True,
    display_init_position: bool = False,
    success_reward: float = 1.0,
    illegal_penalty: float = -4.0,
    max_steps: int = 50,
    **kwargs,
):
    # setup environment
    if maze_name == "u_maze":
        maze = u_maze(
            width,
            height,
            kwargs.get("obstacle_width", width - 3),
            kwargs.get("obstacle_height", height - 4),
        )
        valid_goals = np.array([[height - 2, 1]])
        start_position = kwargs.get("start_position", (1, 1))
    elif maze_name == "t_maze":
        thick = kwargs.get("thick", 1)
        maze = t_maze((width - 2 - thick, height - 2 - thick), thick)
        valid_goals = np.array([[height - 2, width // 2]])
        start_position = kwargs.get("start_position", (1, 1))
    elif maze_name == "double_t_maze":
        maze = double_t_maze()
        valid_goals = np.array([[8, 6]])
        start_position = kwargs.get("start_position", (1, 1))
    elif maze_name == "random_shape_maze":
        maze = random_shape_maze(
            width,
            height,
            kwargs.get("max_shapes", 5),
            kwargs.get("max_size", (width // 2) * (height // 2)),
            kwargs.get("allow_overlap", True),
        )
        empty_postions = np.argwhere(maze == 0).tolist()
        goal_position, start_position = random.sample(empty_postions, 2)
        valid_goals = np.array([goal_position])
    elif maze_name == "random_maze":
        maze = random_maze(
            width, height, kwargs.get("complexity", 0.75), kwargs.get("density", 0.75)
        )
        empty_postions = np.argwhere(maze == 0).tolist()
        goal_position, start_position = random.sample(empty_postions, 2)
        valid_goals = np.array([goal_position])
    else:
        raise ValueError(f"unknown maze name: {maze_name}")

    def describe_function(
        maze: np.ndarray,
        cur_position: Tuple[int, int] = None,
        goal_position: Tuple[int, int] = None,
        init_position: Tuple[int, int] = None,
        move_history: List[str] = None,
    ):
        return describe_observation(
            maze,
            cur_position,
            goal_position,
            init_position,
            display_position,
            display_init_position,
        )

    def reward_function(
        action: str,
        goal_position: Tuple[int, int] = None,
        cur_position: Tuple[int, int] = None,
        possible_actions: Dict[str, Tuple[int, int]] = None,
    ):
        # modify reward calculation
        if cur_position[0] == goal_position[0] and cur_position[1] == goal_position[1]:
            return success_reward
        else:
            return 0

    env = MazeEnv(
        maze=maze,
        valid_goals=valid_goals,
        actions=manhatten_actions,
        max_steps=max_steps,
        display_initial_position=display_init_position,
        describe_function=describe_function,
        reward_function=reward_function,
    )
    observation = env.reset(options={"init_position": start_position})[0].text
    return env, observation, start_position


def maze2str(maze: np.ndarray):
    assert len(maze.shape) == 2
    str_maze = ""
    for line in maze:
        str_maze += "".join(map(lambda x: "囗" if x else "  ", line)) + "\n"
    return str_maze[:-1]
