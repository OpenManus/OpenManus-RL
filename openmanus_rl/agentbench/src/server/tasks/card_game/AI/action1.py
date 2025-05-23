from typing import List, Tuple
import random

from sdk.ai_client import Action, AIClient, Game
from prompt.en import enemy_fish, your_fish

class AI(AIClient):
    def __init__(self, stage) -> None:
        super().__init__()
        self.stage = stage
        self.name_to_id = {"spray": 1, "flame": 2, "eel": 3, "sunfish": 4, "barracuda": 5, "mobula": 6, "octopus": 8, "whiteshark": 9, "hammerhead": 10}
        self.id_to_name = {}
        for name, id in self.name_to_id.items():
            self.id_to_name[id] = name
        self.id_to_name[-1] = "unknown"
        self.ai_fish = []
        random.seed(42)

    def Pick(self, game: Game) -> List[int]:
        pick_list = []     
        self.pos_to_name = enemy_fish
        
        for i in range(4):
            pick_list.append(self.name_to_id[self.pos_to_name[i]])
            self.ai_fish.append(self.name_to_id[your_fish[i]])
            
        #random.shuffle(pick_list)
        return pick_list

    def Assert(self, game: Game) -> Tuple[int, int]:
        if self.stage == 1:
            return (-1, -1)
        else:
            return (self.get_enemy_living_fishes()[0], random.choice(self.ai_fish))

    def Act(self, game: Game) -> Action:
        action = Action(game)
        my_pos: int = self.get_my_living_fishes()[0]
        action.set_action_fish(my_pos)
        return self.auto_valid_action(my_pos, action)
