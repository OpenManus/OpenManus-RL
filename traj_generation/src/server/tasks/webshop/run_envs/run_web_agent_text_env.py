"""
Test the text gym environment.

TODO: move to testing dir for more rigorous tests
"""
import datetime
import os
import sys

from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import *

model_exec = ""


class InteractionLog:
    def __init__(self, file, name) -> None:
        self.file = file
        self.name = name
        self.suffix_index = 0
        while os.path.exists(self.file_name):
            self.suffix_index += 1
        self.stdout = None
        self.logfile = None

    @property
    def file_name(self):
        return self.file + "-" + str(self.suffix_index) + ".log"

    def __enter__(self):
        self.logfile = open(self.file_name, 'w', encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self.logfile
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.logfile.close()
        self.logfile = None
        self.stdout = None


def worker(log_file, idx, rnge):
    with InteractionLog(log_file, idx):
        env = WebAgentTextEnv(observation_mode="text", human_goals=True)
        print("total goals:", len(env.server.goals))
        print("ranging:", rnge)
        scores = []
        for i in range(*rnge):
            env.reset(i)
            print(f"=== Episode #{i} ===")

            policy = eval(model_exec)

            observation = env.observation
            for j in range(100):
                print(observation)
                available_actions = env.get_available_actions()
                print('Available actions:', available_actions)
                action = policy.forward(observation, available_actions)
                if not action:
                    reward = 0
                    break
                observation, reward, done, info = env.step(action)
                print(f'Taking action "{escape(action)}" -> Reward = {reward}')
                if done:
                    break
            else:
                reward = 0
            print(f"#{i} {reward}")
            scores.append(reward)

        print(f"#Average: {sum(scores) / len(scores)}")


if __name__ == '__main__':
    # env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
    arg_length = len(sys.argv)
    if arg_length == 1:
        ranging = (0, 12087)
    elif arg_length == 2:
        ranging = (int(sys.argv[1]), int(sys.argv[1]) + 1)
    elif arg_length == 3:
        ranging = (int(sys.argv[1]), int(sys.argv[2]))
    else:
        ranging = (0, 12087)
    model_exec = input(">>> ")
    print("got EXEC", model_exec)
    log_file = "logs/%s" % (datetime.datetime.now().strftime("%Y-%m-%d=%H-%M-%S"))
    worker(log_file, 0, ranging)
