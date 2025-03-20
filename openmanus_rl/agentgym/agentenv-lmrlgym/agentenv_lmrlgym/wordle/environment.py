import os
from LLM_RL.environment import Text
from llm_rl_scripts.wordle.env.env import ReformatWordleEnvironment, WordleEnvironment
from llm_rl_scripts.wordle.env.game import Vocabulary

vocab_base = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "lmrlgym",
        "llm_rl_scripts",
        "wordle",
        "vocab",
    )
)


class Lmrlgym_WordleEnv:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}

    def create(self):
        idx = self._max_id
        try:
            # vocab = Vocabulary.from_file(os.path.join(vocab_base, f"{vocab_file}.txt"))
            vocab = Vocabulary.from_file(os.path.join(vocab_base, "tweet_words.txt"))
            new_env = ReformatWordleEnvironment(WordleEnvironment(vocab))
            payload = {"id": self._max_id}
            self.env[idx] = new_env
            self.info[idx] = {"done": False, "reward": 0, "deleted": False}
            print(f"-------Env {idx} created--------")
            self._max_id += 1
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            history = (
                Text("This Text is a place holder and will not be used.", False),
                Text(action, True),
            )
            ob, reward, done = self.env[idx].step(history)
            # modify score from [-1, 0] to [0, 1]
            if reward <= 0:
                reward += 1
            ob = ob[-1].text
            if len(ob.strip()) == 0:
                ob = "invalid word"
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

    def reset(self, idx: int, seed: int):
        try:
            self._check_id(idx, True)
            self.env[idx].reset(seed)
            ob = "Let's start Wordle!"
            payload = {"id": idx, "observation": ob}
            self.info[idx].update(
                {
                    "seed": seed,
                    "observation": ob,
                    "done": False,
                    "reward": 0,
                    "deleted": False,
                }
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_filtered_vocab(self, idx: int):
        try:
            self._check_id(idx)
            return str(self.env[idx].env.state.vocab).split("\n")
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
            raise NameError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise NameError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise NameError(f"The task with environment {idx} has finished.")


wordle_server = Lmrlgym_WordleEnv()
