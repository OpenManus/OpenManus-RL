from typing import List, Tuple
import time, requests, json, re, random
from sdk.ai_client import Action, AIClient, Game
from prompt.en import ACTION_DESCRIPTION, GUESS_DESCRIPTION, your_fish, enemy_fish, target_friend, action_prompt, guess_prompt

import os
import logging
import sys

class Agent(AIClient):
    def __init__(self, client, stage, order, save_dir) -> None:
        super().__init__()
        
        self.client = client
        self.stage = stage
        self.order = order
        self.save_dir = save_dir
        
        self.history = []
        self.assert_history = []
        
        self.name_to_id = {"spray": 1, "flame": 2, "eel": 3, "sunfish": 4, "barracuda": 5, "mobula": 6, "octopus": 8, "whiteshark": 9, "hammerhead": 10}
        self.id_to_name = {}
        for name, id in self.name_to_id.items():
            self.id_to_name[id] = name
        self.id_to_name[-1] = "unknown"
        
        # add "0": 1, "1": 2, "2": 3,
        for i in range(11):
            self.name_to_id[str(i)] = i+1

        self.action_type = {'normal': 0, 'active': 1}
        self.skill_type = {'skill_type.aoe': 'AOE', 'skill_type.infight': 'Infight', 'skill_type.crit': 'Crit', 'skill_type.subtle': 'Subtle', 'skill_type.normalattack': 'Normal'}
        self.passive_type = {'passive_type.counter': 'Counter', 'passive_type.deflect': 'Deflect', 'passive_type.reduce': 'Reduce', 'passive_type.heal': 'Heal', 'passive_type.explode': 'Explode'}
        
        logging.basicConfig(format='%(asctime)s %(message)s',filename='../log.txt', level=logging.INFO)
        
        self.known_enemy = []
        self.guess_try_times = 5
        self.action_try_times = 5
        self.died = False

    def Pick(self, game: Game) -> List[int]:
        # TODO: modify for dynamic setting   
        pick_list = []     
        self.name_to_pos = {}
        self.pos_to_name = your_fish
        
        for i in range(4):
            self.name_to_pos[self.pos_to_name[i]] = i
            # add "0": 1, "1": 2, "2": 3,
            self.name_to_pos[str(i)] = i+1
            pick_list.append(self.name_to_id[self.pos_to_name[i]])

        #logging.info("pick_list: " + str(pick_list) + "\n")
        return pick_list

    def _guess_verify(self, move):
        try:
            move['target_position'] = str(move['target_position'])
            target_position = int(move['target_position'])
            guess_type = move['guess_type']

            # verify output
            assert guess_type in enemy_fish # TODO:
            assert int(target_position) in self.get_enemy_living_fishes()

            return True

        except:
            return False

    def _decode_guess(self, output):
        pattern = r"\{[\w\W]*?\}"
        results = [res.replace('\'', '"') for res in re.findall(pattern, output)]

        if results:
            for res in results:
                try:
                    move = json.loads(res)
                except:
                    continue

                if self._guess_verify(move):
                    # self.debug_msg(str(output))
                    # self.debug_msg(str(move))
                    return True, move

        return False, {}

    # 取出目标的position list
    def _non_zero_indexes(self, lst):
        result = []
        for i in range(len(lst)):
            if lst[i] != 0:
                result.append(i)
        return result

    def _guess(self, game):
        for ix in range(self.guess_try_times):
            system = GUESS_DESCRIPTION[self.stage]
            history = self.assert_history#[-3:]

            enemy_action = game.enemy_action
            my_action = game.my_action
            my_assert = game.my_assert

            if enemy_action.action_fish != -1:
                enemy_action_str = {
                    'ACTION_FISH': str(enemy_action.action_fish),
                    'ATK': str(enemy_action.enemy_expected_injury[0]),
                    'TARGET': str(self._non_zero_indexes(enemy_action.enemy_targets)),
                    'SKILL_TYPE': self.skill_type[str(enemy_action.type)]
                }
            else:
                enemy_action_str = {
                    'ACTION_FISH': 'None',
                    'ATK': 'None',
                    'TARGET': 'None',
                    'SKILL_TYPE': 'None'
                }

            trigger_passive = {}
            for _pos, _type in zip(my_action.enemy_passives_id, my_action.enemy_types):
                trigger_passive['Position: ' + str(_pos)] = self.passive_type[str(_type)]

            for _pos, _type in zip(enemy_action.friend_passives_id, enemy_action.friend_types):
                trigger_passive['Position: ' + str(_pos)] = self.passive_type[str(_type)]

            live_enemy = self.get_enemy_living_fishes()
            live_enemy.sort()
            live_enemy = [str(i) for i in live_enemy if not i in self.known_enemy]

            prompt = guess_prompt % (my_assert.assertResult, live_enemy, enemy_action_str, json.dumps(trigger_passive, ensure_ascii=False))

            if self.died:
                return -1, -1

            output = self.client.llm_call(history, prompt, system)
            if output == "### LLM ERROR EXIT ###":
                print("exiting")
                self.died = True
                return -1, -1

            # decode output
            success, move = self._decode_guess(output)
            if success:
                guess_type = self.name_to_id[move['guess_type']]
                target = int(move['target_position'])

                self.known_enemy.append(target)
                self.assert_history.append((prompt, output))
                with open(f'{self.save_dir}/guess_process_{self.order}.jsonl', 'a+') as f:
                    f.write(json.dumps({'try_times': ix, 'cot': output, 'move': move}, ensure_ascii=False) + '\n')

                return (target, guess_type)

        return (-1, -1)

    def Assert(self, game: Game) -> Tuple[int, int]:
        if self.stage == 1:
            return (-1, -1)
        else:
            return self._guess(game)

    def _get_current_state(self, game: Game):
        my_fish = [{'pos': pos, 'id': abs(self.get_my_id(pos)), 'hp': self.get_my_hp(pos), 'atk': self.get_my_atk(pos)} for pos in range(4)]

        enemy_fish = [{'pos': pos, 'id': self.get_enemy_id(pos), 'hp': self.get_enemy_hp(pos), 'atk': 'unknown'} for pos in range(4)]

        state = {"You": [], "Enemy": []}

        ret = "You\nPOSITION,ID,HP,ATK,\n"
        for fish in my_fish:
            x = {}
            x["POSITION"] = str(fish["pos"])
            x["ID"] = self.id_to_name[fish["id"]]
            x["HP"] = str(fish["hp"])
            x["ATK"] = str(fish["atk"])
            state["You"].append(x)

            for item in x:
                ret += str(x[item]) + ","
            ret += "\n"

        ret += "Enemy\nPOSITION,ID,HP,ATK,\n"
        for fish in enemy_fish:
            x = {}
            x["POSITION"] = str(fish["pos"])
            x["ID"] = self.id_to_name[fish["id"]]
            x["HP"] = str(fish["hp"])
            x["ATK"] = str(fish["atk"])
            state["Enemy"].append(x)

            for item in x:
                ret += str(x[item]) + ","
            ret += "\n"

        return json.dumps(state, ensure_ascii=False), ret

    def _move_verfiy(self, move):
        try:
            # TODO: specify 
            pick_fish = move['pick_fish']
            action = move['action']
            assert action in ['normal', 'active']

            move['target_position'] = str(move['target_position'])
            target_position = int(move['target_position'])

            # verify if choose dead fish
            pos = self.name_to_pos[pick_fish]
            assert pos in self.get_my_living_fishes()

            if action == 'active':
                # verify target fish
                if pick_fish == 'spray' or pick_fish == 'eel':
                    move['target_position'] = '0'
                elif pick_fish == "flame" or pick_fish == "sunfish":
                    assert target_position in self.get_my_living_fishes()
                    assert target_position != pos
                elif pick_fish == "mobula" or pick_fish == "octopus":
                    assert target_position in self.get_my_living_fishes()
                elif pick_fish == "barracuda":
                    assert target_position in self.get_enemy_living_fishes()
                elif pick_fish == "whiteshark" or pick_fish == "hammerhead":
                    assert target_position in self.get_enemy_living_fishes() # FIXME: check for lowerest HP
                else:
                    raise
            else:
                assert target_position in self.get_enemy_living_fishes()

            return True

        except:
            return False

    def _decode_move(self, output):
        status = 2  # validation failed

        pattern = r"\{[\w\W]*?\}"
        results = [res.replace('\'', '"') for res in re.findall(pattern, output)]

        if results:
            for res in results:
                try:
                    move = json.loads(res)
                except:
                    continue

                status = 1  # invalid action

                if self._move_verfiy(move):
                    # self.debug_msg(str(output))
                    # self.debug_msg(str(move))
                    return 0, move

        return status, {}

    def _decide(self, game):
        results = []
        for ix in range(self.action_try_times):
            system = ACTION_DESCRIPTION[self.stage]
            history = self.history#[-2:]

            enemy_action = game.enemy_action
            my_action = game.my_action

            if enemy_action.action_fish != -1:
                enemy_action_str = {
                    'ACTION_FISH': str(enemy_action.action_fish),
                    'ATK': str(enemy_action.enemy_expected_injury[0]),
                    'TARGET': str(self._non_zero_indexes(enemy_action.enemy_targets))
                }
            else:
                enemy_action_str = {
                    'ACTION_FISH': 'None',
                    'ATK': 'None',
                    'TARGET': 'None'
                }

            if my_action.action_fish != -1:
                my_action_str = {
                    'ACTION_FISH': str(my_action.action_fish),
                    'ATK': str(my_action.enemy_expected_injury[0]),
                    'TARGET': str(self._non_zero_indexes(my_action.enemy_targets))
                }
            else:
                my_action_str = {
                    'ACTION_FISH': 'None',
                    'ATK': 'None',
                    'TARGET': 'None'
                }
            current_state, current_str = self._get_current_state(game)
            prompt = action_prompt % (current_state, my_action_str, enemy_action_str)

            if self.died:
                return False, ''

            output = self.client.llm_call(history, prompt, system)
            if output == "### LLM ERROR EXIT ###":
                print("exiting")
                self.died = True
                return False, ''

            # decode output
            success, move = self._decode_move(output)

            if success == 0:
                act = Action(game)
                pick_fish = move['pick_fish']
                action_fish_pos = self.name_to_pos[pick_fish]
                action_type = self.action_type[move['action']]
                target = int(move['target_position'])

                try:
                    assert act.set_action_fish(action_fish_pos) == 6
                    assert act.set_action_type(action_type) == 6

                    # active skill is used on ally
                    if action_type == 1 and pick_fish in target_friend:
                        assert act.set_friend_target(target) == 6
                    else:
                        assert act.set_enemy_target(target) == 6
                except:
                    continue

                self.history.append((prompt, output))
                with open(f'{self.save_dir}/thinking_process_{self.order}.jsonl', 'a+') as f:
                    f.write(json.dumps({'try_times': ix, 'cot': output, 'move': move}, ensure_ascii=False) + '\n')

                return True, act
            else:
                results.append(success)

        self.client.send_message("#[ERROR]" + str(min(results)))

        return False, ''

    def Act(self, game: Game) -> Action:

        # ChatGPT Output
        action_success, act = self._decide(game)
        # manually raise to avoid kernel crack
        #if not action_success:
        #    raise 

        return act
