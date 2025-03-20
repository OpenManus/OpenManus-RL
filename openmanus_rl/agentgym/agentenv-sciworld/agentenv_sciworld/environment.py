from scienceworld import ScienceWorldEnv
import uuid


class SciWorldEnv:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.games = []

    def create(self):
        try:
            idx = self._max_id
            self.env[idx] = ScienceWorldEnv()
            self.info[idx] = {"deleted": False, "done": False}
            self._max_id += 1

            exceptions = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}

            for key, value in self.env[idx].tasks.items():
                if key not in exceptions:
                    self.games += [
                        {"taskName": value, "variationIdx": i}
                        for i in range(self.env[idx].getMaxVariations(value))
                    ]
            print(f"-------Env {idx} created--------")
            return {"id": idx}
        except Exception as e:
            return {"error": str(e)}

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            ob, reward, done, info = self.env[idx].step(action)
            payload = {
                "observation": ob,
                "reward": reward,
                "score": info["score"],
                "done": done,
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def reset(self, idx: int, data_idx: int):
        try:
            self._check_id(idx, True)
            self.env[idx].load(
                self.games[data_idx]["taskName"], self.games[data_idx]["variationIdx"]
            )

            task_description = self.env[idx].getTaskDescription()
            ob, reward, done, info = self.env[idx].step("look around")

            payload = {
                "task_name": self.games[data_idx]["taskName"],
                "var_num": self.games[data_idx]["variationIdx"],
                "task_description": task_description,
                "observation": ob,
                "reward": reward,
                "score": info["score"],
                "deleted": False,
                "done": done,
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_action_hint(self, idx: int):
        try:
            self._check_id(idx)
            return {
                "possible_actions": self.env[idx].getPossibleActions(),
                "possible_objects": self.env[idx].getPossibleObjects(),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_goals(self, idx: int):
        try:
            self._check_id(idx)
            return {"goals": self.env[idx].getGoalProgressStr()}
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise ValueError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise ValueError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise ValueError(f"The task with environment {idx} has finished.")


server = SciWorldEnv()
