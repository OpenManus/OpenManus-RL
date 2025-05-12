import json
import re
import numpy as np
import textworld
from textworld import Agent
from llm_rl_scripts.text_nav.env import TextNavEnv


walkthrough = {
    "Bedroom": [["go west"], ["go west"]],
    "Office": [["go north"], ["go north"]],
    "Bathroom": [["go north"], ["go north"]],
    "Living Room": [["take stale food from table", "go west"], ["go west"]],
    "Kitchen": [["go east"], ["open fridge", "insert stale food into fridge", "close fridge"]],
    "Dining Room": [["go east"], ["go west"]],
    "Garden": [["go south"], ["go south"]],
    "Backyard": [["go east"], ["go east"]],
}


class PartialWalkthroughAgent(Agent):
    """ Agent that behaves optimally for subset of rooms and randomly the rest. """

    def __init__(self, optimal_rooms = ["Bedroom", "Office", "Bathroom", "Living Room"]):
        self.optimal_rooms = optimal_rooms
        self.rng = np.random.RandomState(1234)

    def reset(self, env):
        env.infos.admissible_commands = True
        env.display_command_during_render = True
        
        self.has_stale_food = False
        self.commands = iter([])

    def act(self, game_state, reward, done):
        p = re.compile("-= (.*) =-")
        room = p.search(game_state.description).group(1)

        try:
            action = next(self.commands)
        except StopIteration:
            if self.rng.rand() > 0.9 and room in self.optimal_rooms:
                idx = int(self.has_stale_food)
                self.commands = iter(walkthrough[room][idx])
                action = next(self.commands)
            else:
                action = self.rng.choice(game_state.admissible_commands)

        action = action.strip()  # Remove trailing \n, if any.
        if action == "take stale food from table": 
            self.has_stale_food = True
        return action


if __name__ == "__main__":
    max_nb_steps = 15

    data = []
    for i in range(200):
        if i % 20 == 0:
            print("{}/200".format(i))

        env = TextNavEnv(display_location=True, display_inventory=True)

        if np.random.rand() > 0.25:
            optimal_rooms = ["Office", "Bathroom", "Garden", "Backyard"]
        else:
            optimal_rooms = ["Kitchen", "Dining Room", "Living Room", "Bedroom"]
        agent = PartialWalkthroughAgent(optimal_rooms)
        agent.reset(env.env)
        env._reset()
    
        text_history = [env.state.feedback.strip()]
        rewards = []
        dones = []
    
        for _ in range(max_nb_steps):
            command = agent.act(env.state, env.state.score, env.state.done)
            env._step(command)
    
            text_history.append(command)
            text_history.append(env.state.feedback.strip())
            rewards.append(env.state.score)
            dones.append(env.state.done)
            if env.state.done:
                break
        env.env.close()

        dones[-1] = True

        data.append({"text_history": text_history, 
                     "rewards": rewards,
                     "dones": dones,})


    with open("bc_full_info.json", "w") as f:
        json.dump(data, f)

