from typing import List, Tuple
import random

from sdk.ai_client import Action, AIClient, Game
from random import randint
from prompt.en import enemy_fish, your_fish

import os
import logging
import sys

class AI(AIClient):
    def __init__(self, stage) -> None:
        super().__init__()
        self.stage = stage
        self.guessed = [[], [], [], []]
        self.last_fish = -1
        self.last_type = -1
        
        self.name_to_id = {"spray": 1, "flame": 2, "eel": 3, "sunfish": 4, "barracuda": 5, "mobula": 6, "octopus": 8, "whiteshark": 9, "hammerhead": 10}
        self.id_to_name = {}
        for name, id in self.name_to_id.items():
            self.id_to_name[id] = name
            
        self.skill_type = {'skill_type.aoe': 'AOE', 'skill_type.infight': 'Infight', 'skill_type.crit': 'Crit', 'skill_type.subtle': 'Subtle', 'skill_type.normalattack': 'Normal'}
        self.passive_type = {'passive_type.counter': 'Counter', 'passive_type.deflect': 'Deflect', 'passive_type.reduce': 'Reduce', 'passive_type.heal': 'Heal', 'passive_type.explode': 'Explode'}
        self.clue = {"spray": ["AOE", "Counter"], "flame": ["Infight", "Counter"], "eel": ["AOE", "Deflect"], "sunfish": ["Infight", "Deflect"], 
                     "barracuda": ["Crit", "Reduce"], "mobula": ["Subtle", "Reduce"], "octopus": ["Subtle", "Heal"], "whiteshark": ["Crit", "Heal"],
                     "hammerhead": ["Crit", "Explode"]}
        logging.basicConfig(format='%(asctime)s %(message)s',filename='../action2.txt', level=logging.INFO)
        random.seed(42)

    def Pick(self, game: Game) -> List[int]:
        pick_list = []     
        self.pos_to_name = enemy_fish
        
        for i in range(4):
            pick_list.append(self.name_to_id[self.pos_to_name[i]])
        #random.shuffle(pick_list)
        return pick_list
    
    def add_possible(self, fish, type):
        if self.get_enemy_id(fish) != -1:
            return
        self.guessed[fish].append(type)

    def ass(self, fish, type):
        # logging.info(str((fish, type)) + "\n")
        self.last_fish = fish
        self.last_type = type
        return (fish, type)
    
    def Assert(self, game: Game) -> Tuple[int, int]:
        if self.stage == 1:
            return (-1, -1)
        else:
            # 如果在需要 assert 的列表中
            enemy_action = game.enemy_action
            my_action = game.my_action
            # my_assert = game.my_assert
            # if my_assert.assertResult == False:
            #    self.guessed[self.last_fish].append(self.last_type)
                
            live_enemy = self.get_enemy_living_fishes()
            live_enemy.sort()
            
            # 如果上次行动的鱼其实是未知的
            if self.get_enemy_id(enemy_action.action_fish) == -1:
                # 如果返回目标很多 代表有 aoe
                if enemy_action.is_skill == True:
                    assert enemy_action.type != "normalattack"
                    self.add_possible(enemy_action.action_fish, enemy_action.type)

                # 判断可能的被动
                l = zip(enemy_action.friend_passives_id + my_action.enemy_passives_id, enemy_action.friend_types + my_action.enemy_types)
                for _pos, _type in l:
                    self.add_possible(_pos, self.passive_type[str(_type)])
            
            # 如果完全确定了         
            for i in range(4):
                for (id, name) in self.id_to_name.items():
                    if set(self.clue[name]).issubset(self.guessed[i]):
                        return self.ass(i, id)
                
            # 在剩下的鱼中随机选择一个乱猜
            know_pos, know_type = [0, 1, 2, 3], list(self.id_to_name.keys())
            for i in range(4):
                if self.get_enemy_id(i) != -1 or live_enemy.count(i) == 0: # 如果已经确定了那么就删掉
                    know_pos.remove(i)
                    if self.get_enemy_id(i) != -1:
                        know_type.remove(self.get_enemy_id(i))
            # logging.info(str(know_pos) + " know_pos\n")
            # logging.info(str(know_type) + " know_type\n")
            if len(know_pos) == 0:
                return (-1, -1)
            else:
                return self.ass(random.choice(know_pos), random.choice(know_type))
        
    def get_type(self, fish):
        return self.id_to_name[abs(self.get_my_id(fish))]
    
    def get_specific(self, my_alive, types):
        for type in types:
            for fish in my_alive:
                if self.get_type(fish) == type:
                    return fish
        return -1
    
    #策略 选择敌方
    def Act(self, game: Game) -> Action:
        action = Action(game)
        enemy_action_fish = game.enemy_action.action_fish
        
        # 如果友方可以致命一击普通攻击敌方，选择直接攻击
        my_alive = self.get_my_living_fishes()
        enemy_alive = self.get_enemy_living_fishes()
        max_atk_pos = -1
        max_atk = 0
        for my_pos in my_alive:
            if max_atk < self.get_my_atk(my_pos):
                max_atk = self.get_my_atk(my_pos)
                max_atk_pos = my_pos
            
        enemy_pos = self.get_lowest_health_enemy()
        normal_atk = self.get_my_atk(max_atk_pos) * 0.5

        try:
            if normal_atk >= self.get_enemy_hp(enemy_pos):
                assert action.set_action_fish(max_atk_pos) == 6
                assert action.set_action_type(0) == 6
                assert action.set_enemy_target(enemy_pos) == 6
                return action
            
            # 如果可以使用 aoe
            if len(enemy_alive) >= 2:
                fish = self.get_specific(my_alive, ["spray", "eel"])
                if fish != -1:
                    action.set_action_fish(fish)
                    action.set_action_type(1)
                    action.set_enemy_target(0)
                    return action
            
            # 如果可以使用 Crit
            fish = self.get_specific(my_alive, ["barracuda", "whiteshark", "hammerhead"])
            lowest_hp = self.get_lowest_health_enemy()
            if fish != -1:
                assert action.set_action_fish(fish) == 6
                assert action.set_action_type(1) == 6
                assert action.set_enemy_target(lowest_hp) == 6    
                return action
        
            assert action.set_action_fish(max_atk_pos) == 6
            assert action.set_action_type(0) == 6
            
            # 攻击上一轮攻击我方的鱼
            if enemy_action_fish in enemy_alive:
                assert action.set_enemy_target(enemy_action_fish) == 6
            else:
                assert action.set_enemy_target(enemy_alive[0]) == 6
        
        except:
            logging.error("find\n")        
        # 否则攻击敌方攻击力最高的
        # target = 0
        # target_atk = -1
        # for fish in enemy_alive:
        #     if self.enemy_atk_record[fish] > target_atk:
        #         target = fish
        #         target_atk = self.enemy_atk_record[fish]
        
        # action.set_enemy_target(target)
        
        return action