from .utils import setup_maze_env
from LLM_RL.environment import Text


class Lmrlgym_MazeEnv:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.available_actions = ["move left", "move right", "move up", "move down"]

    def create(self):
        idx = self._max_id
        try:
            payload = {
                "id": self._max_id,
            }
            self.info[idx] = {
                "done": False,
                "reward": 0,
                "deleted": False,
            }
            print(f"-------Env {idx} created--------")
            self._max_id += 1
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            history = (Text(action + "\n", True),)
            ob, reward, done = self.env[idx].step(history)
            ob = ob[-1].text
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[idx].update(
                {
                    "observation": ob,
                    "reward": self.info[idx]["reward"] + reward,
                    "done": done,
                }
            )
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, idx: int, game: int, **kwargs):
        try:
            self._check_id(idx, True)
            start_positions = [
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 7),
                (1, 8),
                (1, 9),
                (1, 10),
                (1, 11),
                (2, 3),
                (3, 3),
                (4, 3),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (6, 6),
                (7, 6),
                (5, 7),
                (5, 8),
                (5, 9),
                (4, 9),
                (3, 9),
                (2, 9),
                (8, 6),
            ]
            if game >= len(start_positions):
                raise NameError(f"The game {game} is not valid.")
            self.env[idx], ob, init_position = setup_maze_env(
                "double_t_maze", start_position=start_positions[game], **kwargs
            )
            ob = (
                self.env[idx]
                .reset(
                    options={"goal": self.env[idx].goal, "init_position": init_position}
                )[0]
                .text
            )
            payload = {"observation": ob}
            self.info[idx] = {
                "maze_name": "double_t_maze",
                "observation": ob,
                "done": False,
                "reward": 0,
                "deleted": False,
                "goal_position": self.env[idx].goal,
                "init_position": init_position,
            }
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_available_actions(self):
        return self.available_actions

    def get_detailed_info(self, idx: int):
        try:
            self._check_id(idx)
            payload = self.info[idx].copy()
            del payload["goal_position"]
            del payload["init_position"]
            return payload
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise NameError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise NameError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise NameError(f"The task with environment {idx} has finished.")


maze_server = Lmrlgym_MazeEnv()
