from .environment import TextCraftEnv
from .crafting_tree import CraftingTree


class TextCraft_Wrapper:
    def __init__(self, minecraft_dir="agentenv_textcraft/"):
        self._max_id = 0
        self.env = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)

    def create(self, commands: str = None, goal: str = None):
        try:
            id = self._max_id
            new_env = TextCraftEnv(
                crafting_tree=self.crafting_tree, commands=commands, goal=goal
            )
            ob, _ = new_env.reset(data_idx=id)
            print(f"-------Env {id} created--------")
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.env[id] = new_env
            self.info[id] = {
                "observation": ob,
                "done": False,
                "reward": 0,
                "deleted": False,
            }
            self._max_id += 1
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def step(self, id: int, action: str):
        try:
            self._check_id(id)
            (ob, reward, done, _, _) = self.env[id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, id: int, data_idx: int):
        try:
            self._check_id(id)
            ob, _ = self.env[id].reset(data_idx=data_idx)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.info[id].update(
                {"observation": ob, "done": False, "reward": 0, "deleted": False}
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, id: int):
        if id not in self.info:
            raise NameError(f"The id {id} is not valid.")
        if self.info[id]["deleted"]:
            raise NameError(f"The task with environment {id} has been deleted.")

server = TextCraft_Wrapper()
