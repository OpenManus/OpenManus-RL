import gym
import os
import numpy as np
import gymnasium
import matplotlib.pyplot as plt

class BabyAI(gym.Env):
    def __init__(self, 
                 max_episode_steps=50, 
                 game_name="BabyAI-GoToRedBall-v0",
                 seed=1234,
                 game_config=None,
                 render_path="temp/babyai_render",
                 need_render=False,
                 obs_to_reward=None,
                 difficulty="easy",
                 ):
        
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.error_message = {}
        self.game_name = game_name
        self.seed = seed
        self.game_config = game_config
        
        self.env = gymnasium.make(game_name)
        self.render_path = render_path
        self.need_render = need_render
        self.obs_to_reward = obs_to_reward
        self.store_all_obs_to_reward = obs_to_reward
        self.difficulty = difficulty
        if self.obs_to_reward is not None:
            if isinstance(self.obs_to_reward[0], list):
                self.num_obs_to_reward = len(self.obs_to_reward[0])
            else:
                self.num_obs_to_reward = len(self.obs_to_reward)
        self.reset()
        
        
    def render(self, mode='human'):
        if not os.path.exists(self.render_path):
            os.makedirs(self.render_path)  
        
        rgb_img = self.env.unwrapped.get_frame(
            highlight=self.env.unwrapped.highlight, tile_size=self.env.unwrapped.tile_size
        )
        
        output_path = os.path.join(self.render_path, f"step_{self.steps}.png")
        plt.imsave(output_path, rgb_img)
        
        return output_path
        
    
    def _get_info(self):
        return self.infos
    
    def _get_obs(self):
        return self.states[-1]
    
    def _get_goal(self):
        return self.goal
    
    def _get_history(self):
        return self.history
    
    def _get_action_space(self):
        return list(self.action_space.keys()) # return a list of valid actions
    
    def _is_done(self):
        return self.done
    
    def match_style(self, obs, pattern):
        pattern = pattern.strip()
        split_token = "**"
        if "**" not in pattern:
            split_token = "*"
        pattern_list = pattern.strip().split(split_token)
        all_obs = obs.split(".")
        for obs_temp in all_obs:
            flag = True
            for p in pattern_list:
                p = p.strip(".")
                if p not in obs_temp:
                    flag = False
            if flag:
                return True
        return False
    
    def update_reward(self, obs):
        if self.obs_to_reward is None:
            return
        if len(self.obs_to_reward) == 0:
            return
        if isinstance(self.obs_to_reward[0], list):
            need_to_award = False   
            path_length = len(self.obs_to_reward[0])
            for i in range(path_length):
                for obs_temp in self.obs_to_reward:
                    if self.match_style(obs, obs_temp[i]):
                        need_to_award = True
                        break
                
                if need_to_award:
                    self.points += 1
                    self.reward = max(self.reward, self.points/self.num_obs_to_reward)
                    for obs_temp in self.obs_to_reward:
                        obs_temp.remove(obs_temp[i])
                    break
                
        else:
            for pattern in self.obs_to_reward:
                if self.match_style(obs, pattern):
                    self.points += 1
                    self.reward = max(self.reward, self.points/self.num_obs_to_reward)
                    self.obs_to_reward.remove(pattern)
                    break
                        
        
    
    def update(self, action, obs, reward, done, infos): # update the environment after taking an action
        for k, v in infos.items():
            self.infos[k] = v
        
       
        self.done = done
        self.history.append(("action", action))
        self.history.append(("reward", reward))
        
        new_obs, new_action_space = self.postprocess_obs(obs)
        
        if self.done:
            new_obs += "\n The task is completed."
        
        if self.obs_to_reward is not None:
            self.update_reward(new_obs)
        else:
            self.reward = reward
            
        if self.done: # in case the model find a way to skip a step
            if self.reward <= 0.5:
                self.done = False
            
        if self.reward == 1 and not self.done:    # if the agent has already reached the goal, but the task is not completed, we fix the error.
            self.done = True
            new_obs += "\n The task is completed."
            
        
        self.history.append(("state", new_obs))
        self.states.append(new_obs)
        
        self.action_space = new_action_space
        
        self.steps += 1
        self.obs_2d = obs["image"] # keep the 2d observation for visualization, also used to double check if a step is implemented correctly
        
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        
        
    def update_info(self, action, info): # update the environment after taking an action, the action is not implemented correctly
        self.history.append(("action", action))
        self.history.append(("reward", self.reward))
        self.history.append(("state", info))
        self.states.append(info)
        self.steps += 1
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
    
    def get_next_pos(self, pos, action, dir): # get the next position after taking an action
        if action == 0:
            dir = (dir-1)%4
            
        elif action == 1:
            dir = (dir+1)%4
        
        elif action == 2:
            dir_vec = DIR_TO_VEC[dir]
            pos = tuple(pos + dir_vec)
            
        return pos, dir
    
    def find_path(self, init_pos, goal, all_objs, all_barriers, init_dir, xrange, yrange, arrive=False): # find the shortest path from pos to goal, all_objs is a list of position of objects, need to avoid them
        all_things = all_objs + all_barriers
        pos = init_pos
        dir = init_dir
        graph = dict()
        queue = [(pos, dir)]
        state = set()
        
        
        while len(queue) > 0:
            pos, dir = queue.pop(0)
            state.add((pos, dir))
            
            if arrive:
                if pos[0]==goal[0] and pos[1]==goal[1]:
                    # get all actions that leas to current state
                    path = []
                    while (pos, dir) != (init_pos, init_dir):
                        (pos, dir), action = graph[(pos, dir)]
                        path.append(action)
                    path = path[::-1]
                    
                    return path
            else:
                if goal[0] - pos[0] == DIR_TO_VEC[dir][0] and goal[1] - pos[1] == DIR_TO_VEC[dir][1]:
                    path = []
                    while (pos, dir) != (init_pos, init_dir):
                        (pos, dir), action = graph[(pos, dir)]
                        path.append(action)
                    path = path[::-1]
                    
                    return path
            
            for action in [2, 0, 1]:
                new_pos, new_dir = self.get_next_pos(pos, action, dir)
                is_obstacle = False
                for obj in all_things:
                    if new_pos[0] not in xrange or new_pos[1] not in yrange:
                        is_obstacle = True
                        break
                    if (new_pos, new_dir) in state:
                        is_obstacle = True
                        break
                    if obj["abs_pos"] == new_pos:
                        if "wall" in obj["name"] or "box" in obj["name"] or "lava" in obj["name"] or "ball" in obj["name"] or "key" in obj["name"]:
                            is_obstacle = True
                            break
                if not is_obstacle:
                    queue.append((new_pos, new_dir))
                    graph[(new_pos, new_dir)] = ((pos, dir), action)
            
        return None

    def postprocess_obs(self, obs): # postprocess the observation, translate the observation into description and possible actions
        
        _, vis_mask = self.env.unwrapped.gen_obs_grid()
        view_size = self.env.unwrapped.agent_view_size
        pos = self.env.unwrapped.agent_pos
        f_vec = self.env.unwrapped.dir_vec
        r_vec = self.env.unwrapped.right_vec
        
        # Compute the absolute coordinates of the top-left corner
        # of the agent's view area
        top_left = pos + f_vec * (view_size - 1) - r_vec * (view_size // 2)
        
        # calculate the range of the absolute coordinates
        vecs = - f_vec + r_vec
        boarders = top_left + view_size*vecs
    
        xboarder = boarders[0]
        if xboarder < top_left[0]:
            xrange = range(xboarder, top_left[0]+1)
        else:
            xrange = range(top_left[0], xboarder)
            
        yboarder = boarders[1]
        if yboarder < top_left[1]:
            yrange = range(yboarder, top_left[1]+1)
        else:
            yrange = range(top_left[1], yboarder)
        
        grid = obs["image"]
        dir = obs["direction"]
        all_objs = []
        
        # identify distance to walls and barriers (box) in four directions
        left_dis = 0
        all_barriers = []
        
        for vis_j in range(0, view_size):
            for vis_i in range(0, view_size):
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                distance = abs(pos[0]-abs_i) + abs(pos[1]-abs_j)
                
                if abs_i < 0 or abs_j < 0:
                    continue
                
                if distance == 0: # in case the agent counts the carrying object as an additional object
                    continue
                
                obj_type = IDX_TO_OBJECT[grid[vis_i, vis_j, 0]]
                obj_color = IDX_TO_COLOR[grid[vis_i, vis_j, 1]]
                obj_state = IDX_TO_STATE[grid[vis_i, vis_j, 2]]
                
                # identify object of interest
                if obj_type in ["door", "key", "ball", "box", "goal", "lava", "wall"]:
                    if obj_type == "door":
                        obj_name = obj_color + " " + obj_state + " " + obj_type
                    else:
                        obj_name = obj_color + " " + obj_type
                    
                    all_objs.append({"name": obj_name, "abs_pos":(abs_i, abs_j), "dis":distance})
                    
                if obj_type in ["box", "wall"]:
                    self_dir = DIR_TO_VEC[dir] # get the direction of the agent
                    obj_relative_pos = (abs_i - pos[0], abs_j - pos[1]) # get the relative position of the object   
                    # check if the object is in front of the agent
                    if np.cross(self_dir, obj_relative_pos) == 0:
                        all_barriers.append({"name": obj_type, "abs_pos":(abs_i, abs_j), "dis":np.dot(self_dir, obj_relative_pos)})
                
        # sort by distance, from near to far
        all_objs.sort(key=lambda x: x["dis"])
        if len(all_objs) > 0:
            cnt_observe = dict()
            obj_description = "In front of you in this room, you can see several objects: "
            for obj_temp in all_objs:
                if 'wall' in obj_temp["name"]:
                    continue
                obj_temp_pos = obj_temp["abs_pos"]
                obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                self_dir = DIR_TO_VEC[dir]
                front_dis = np.dot(self_dir, obj_temp_relative) 
                right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                pos_desc_temp = ""
                
                if right_dis == 0:
                    pos_desc_temp = "right in front of you " + str(int(front_dis)) + " steps away. "
                elif right_dis > 0:
                    pos_desc_temp = str(int(front_dis)) + " steps in front of you and " + str(int(right_dis)) + " steps to your right. "
                else:
                    pos_desc_temp = str(int(front_dis)) + " steps in front of you and " + str(int(-right_dis)) + " steps to your left. "
                
                if obj_temp["name"] not in cnt_observe:
                    cnt_observe[obj_temp["name"]] = 1
                else:
                    cnt_observe[obj_temp["name"]] += 1
                obj_description += "There is a " + obj_temp["name"] + " " + str(cnt_observe[obj_temp["name"]]) + " "+ pos_desc_temp + " "
        else:
            obj_description = "You cannot see any objects within sight."
        
        barrier_description = "The room has walls around you. "
        if len(all_barriers) > 0:
            all_barriers.sort(key=lambda x: x["dis"])
            barrier_dis_pos = all_barriers[0]["dis"]
            
            barrier_description += "You are facing a " + all_barriers[0]["name"] + " " + str(barrier_dis_pos) + " steps away. "
            
        carry_description = ""
        carrying = self.env.unwrapped.carrying
        if carrying is not None:
            carrying_type = carrying.type
            carrying_color = carrying.color
            carry_description = "You are carrying a " + carrying_color + " " + carrying_type + "."
        else:
            carry_description = "You are not carrying anything."
         
        description = obj_description + barrier_description + carry_description
        
        # ---------------------------finish processing the description of the goal--------------------------------
        
        # ---------------------------    process possible actions space     --------------------------------
        
        possible_actions = {"turn left": [0], "turn right": [1]}
        error_message = {} # create finegrained error message for failed to execute actions
        
        # can the agent move forward?
        if len(all_barriers) == 0 or all_barriers[0]["dis"] > 1: # if there is no barrier or the barrier is far away, the agent can move forward
            possible_actions["move forward"] = [2]
        else:
            error_message["move forward"] = "There is a barrier in front of you, you can't move forward."
        
        # go to pickup an object
        if carrying is None:
            if len(all_objs) > 0:
                
                cnt_object = dict()
                for i, obj_temp in enumerate(all_objs):
                    if 'wall' in obj_temp["name"]:
                        continue
                    if 'door' in obj_temp["name"]:
                        continue
                    
                    if 'goal' in obj_temp["name"]:
                        continue
                    
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                    self_dir = DIR_TO_VEC[dir]
                    
                    obj_name = obj_temp["name"]
                    
                    front_dis = np.dot(self_dir, obj_temp_relative) 
                    right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False) 
                    
                    if actions_temp is not None:
                        actions_temp.append(3) # add pickup action at the end
                    
                    
                    
                        if "pickup "+ obj_name+ " " + str(1) not in possible_actions: # note that this action space is not necessarily successful, we will execute the actions step by step and stops if failed.
                            cnt_object[obj_name] = 1
                            possible_actions["pickup "+ obj_name + " " + str(1)] = actions_temp
                        else:
                            cnt_object[obj_name] += 1
                            possible_actions["pickup "+ obj_name + " " + str(cnt_object[obj_name])] = actions_temp
                    else:
                        if "pickup "+ obj_name+ " " + str(1) not in possible_actions:
                            error_message["pickup "+ obj_name + " " + str(1)] = "You cannot pickup " +  obj_name + " " + str(1) + ", as there is no path leading to it."
                        else:
                            error_message["pickup "+ obj_name + " " + str(cnt_object[obj_name]+1)] = "You cannot pickup " + obj_name + " " + str(cnt_object[obj_name]+1) + ", as there is no path leading to it."
                            
        
        # drop an object
        if carrying is not None:
            drop_pos = tuple(pos + DIR_TO_VEC[dir])
            can_drop = True
            for obj_temp in all_objs:
                if obj_temp["abs_pos"] == drop_pos:
                    for obj_type in ["wall", "box", "lava", "ball", "key"]:
                        if obj_type in obj_temp["name"]:
                            can_drop = False
                            break
            if can_drop:
                possible_actions["drop"] = [4]
            
            else:
                error_message["drop"] = "You cannot drop the object, as there is already an object in front of you."
        else:
            error_message["drop"] = "You cannot drop the object, as you are not carrying anything."
        
        # go through a door or toggle a door or toggle a door with a key
        if len(all_objs)>0:
            cnt_door = dict()
            for obj_temp in all_objs:  
                if 'door' in obj_temp["name"]:
                    if obj_temp["name"] not in cnt_door:
                        cnt_door[obj_temp["name"]] = 1
                    else:
                        cnt_door[obj_temp["name"]] += 1
                    
                if 'open door' in obj_temp["name"]:
                    
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                    self_dir = DIR_TO_VEC[dir]
                    
                    obj_name = obj_temp["name"]
                    
                    front_dis = np.dot(self_dir, obj_temp_relative) 
                    right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                    
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange,  arrive=True)
                    if actions_temp is not None:
                        possible_actions["go through "+ obj_temp["name"] + " "+ str(cnt_door[obj_temp["name"]])] = actions_temp
                    else:
                        error_message["go through "+ obj_temp["name"] + " "+ str(cnt_door[obj_temp["name"]])] = "You cannot go through " + obj_temp["name"] + " "+ str(cnt_door[obj_temp["name"]]) + ", as there is no path leading to it."
                
                if 'closed door' in obj_temp["name"]:
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                    self_dir = DIR_TO_VEC[dir]
                    
                    obj_name = obj_temp["name"]
                    
                    front_dis = np.dot(self_dir, obj_temp_relative) 
                    right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                    
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange,  arrive=False)
                    
                    if actions_temp is not None:
                        possible_actions["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = actions_temp + [5, 2]
                    else:
                        error_message["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " "+str(cnt_door) + ", as there is no path leading to it."
                    if actions_temp == []:
                        possible_actions["toggle"] = [5]
                    error_message["go through "+ obj_temp["name"] + " "+ str(cnt_door[obj_temp["name"]])] = "You cannot go through " + obj_temp["name"] + " "+ str(cnt_door[obj_temp["name"]]) + ", as it is closed. You should toggle it first."  
                    
                if 'locked door' in obj_temp["name"]:
                        
                    if carrying is None or carrying.type != 'key':   
                        error_message["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]]) + ", as you are not carrying a key."
                        continue
                    if carrying.color != obj_temp["name"].split(" ")[0]:
                        error_message["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]]) + ", as the color of the key you are carrying does not match the color of door."
                        continue
                
                    
                    obj_temp_pos = obj_temp["abs_pos"]
                    obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                    self_dir = DIR_TO_VEC[dir]
                    
                    obj_name = obj_temp["name"]
                    
                    front_dis = np.dot(self_dir, obj_temp_relative) 
                    right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                    
                    actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange,  arrive=False)
                    
                    if actions_temp is not None:
                        possible_actions["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = actions_temp + [5, 2]
                    else:
                        error_message["toggle and go through " + obj_temp["name"] + " "+str(cnt_door[obj_temp["name"]])] = "You cannot toggle and go through " + obj_temp["name"] + " "+str(cnt_door) + ", as there is no path leading to it."
                    if actions_temp == []:
                        possible_actions["toggle"] = [5]
                        
                        
        # go to the goal
        if len(all_objs) > 0:
            for obj_temp in all_objs:
                if "goal" not in obj_temp["name"]:
                    continue
                
                obj_temp_pos = obj_temp["abs_pos"]
                obj_temp_relative = (obj_temp_pos[0] - pos[0], obj_temp_pos[1] - pos[1])
                self_dir = DIR_TO_VEC[dir]
                
                obj_name = obj_temp["name"]
                
                front_dis = np.dot(self_dir, obj_temp_relative) 
                right_dis = np.dot(DIR_TO_VEC[(dir+1)%4], obj_temp_relative)
                
                actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange,arrive=True)
                if actions_temp is not None:
                    possible_actions["go to goal"] = actions_temp
                else:
                    error_message["go to goal"] = "You cannot go to the goal, as there is no path leading to it."
        
        # go to object
        if len(all_objs) > 0:
            cnt_goto = dict()
            for obj_temp in all_objs:
                if "wall" in obj_temp["name"]:
                    continue
                if "goal" in obj_temp["name"]:
                    continue
                obj_name = obj_temp["name"]
                obj_temp_pos = obj_temp["abs_pos"]
                
                actions_temp = self.find_path(pos, obj_temp_pos, all_objs, all_barriers, dir, xrange, yrange, arrive=False)
                if actions_temp is not None:
                    if "go to " + obj_name + ' 1' not in possible_actions:
                        possible_actions["go to " + obj_name+ ' 1'] = actions_temp
                        cnt_goto[obj_name] = 1
                    else:
                        cnt_goto[obj_name] += 1
                        possible_actions["go to " + obj_name+ ' ' + str(cnt_goto[obj_name])] = actions_temp
                else:
                    if "go to " + obj_name + ' 1' not in possible_actions:
                        error_message["go to " + obj_name+ ' 1'] = "You cannot go to " + obj_name+ ' 1' + ", as there is no path leading to it."
                    else:
                        error_message["go to " + obj_name+ ' ' + str(cnt_goto[obj_name]+1)] = "You cannot go to " + obj_name+ ' ' + str(cnt_goto[obj_name]+1) + ", as there is no path leading to it."
        
        # add check action space as a special action
        possible_actions["check available actions"] = []
        self.error_message = error_message
        return description, possible_actions
                
        
    def reset(self):
        obs, infos = self.env.reset(seed=self.seed)
        if self.store_all_obs_to_reward is not None:
            self.obs_to_reward = self.store_all_obs_to_reward.copy()
        else:
            self.obs_to_reward = None
        self.goal = self.env.unwrapped.mission 
        if "then" in self.goal: 
            self.goal = self.goal.replace("then", "and")
        if "after you" in self.goal:
            self.goal = self.goal.replace("after you", "and")
        
        description, possible_actions = self.postprocess_obs(obs) # postprocess the observation, translate the observation into description and possible actions
        self.action_space = possible_actions # record the possible actions, each action corresponds to a list of low-level actions
        self.init_obs = description
        
        self.infos = infos # record last step info, infos should be an empty dict for babyai :)
        self.states = [self.init_obs]  # record a stream of states
        self.history = [("state", self.init_obs)] # record a stream of s0, a0, r0, s1, a1, r1, ...
        self.steps = 0
        
        self.infos["goal"] = self.goal
        self.infos["states"] = self.states
        self.infos["history"] = self.history
        self.infos["steps"] = self.steps
        self.infos["state"] = self.states[-1]
        
        self.obs_2d = obs["image"] # keep the 2d observation for visualization, also used to double check if a step is implemented correctly
        self.reward = 0
        self.points = 0
        self.done = False
        
    def verify_action(self, action, obs):
        # verify if the action is implemented correctly
        # for convenience, now only implement as if the state is changed after taking an action
        if (obs["image"] != self.obs_2d).sum() > 0:
            return True
        else:
            return False
        
        
    def check_action_is_valid(self, action):
        action_space = self.action_space
        state = self.states[-1]
        if "check" in action:
            return True, None
        if action == "":
            return False, "No change in state."
        if action not in action_space:
            if action in self.error_message:
                return False, self.error_message[action]
            else:
                return False, "The action is not recognized. Please check valid actions."
        else:
            return True, None
    
    def step(self, action):
        action = action.lower()
        action = action.strip()
        is_valid, error = self.check_action_is_valid(action)
        if not is_valid:
            self.update_info(action, error)
            self.infos["action_is_valid"] = False
            return self._get_obs(), self.reward, self.done, self.infos
        elif action == "check available actions" or "check" in action:
            action_info = "You can take the following actions: " + ", ".join(self._get_action_space())
            self.update_info(action, action_info)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done, self.infos
        else:
            action_list = self.action_space[action] # get the list of low-level actions corresponding to the action
            # print(action_list)
            
            if action_list == []:
                self.update_info(action, "No change in state.")
                return self._get_obs(), self.reward, self.done, self.infos
            
            for action_step in action_list:
                obs, reward, done, truncated, infos = self.env.step(action_step) # five returns using the new step API
                if not self.verify_action(action_step, obs):
                    break  # if the action is not implemented correctly, stop taking the next action, and return the current observation
                else:
                    self.obs_2d = obs["image"] # update the 2d observation, as we need to check if the action is implemented correctly at a lower granularity
            # print(self.env)
            self.update(action, obs, reward, done, infos) # update the environment after all the low-level actions are taken
            # print("reward: ", self.reward)
            self.infos["action_is_valid"] = True
            return self._get_obs(), self.reward, self.done, self.infos
    
    def save_log(self, log_path):
        history = self.infos["history"]
        with open(log_path, 'w') as f:
            for item in history:
                item_name = item[0]
                item_content = item[1]
                if item_content is None:
                    continue
                f.write(item_name + ": " + str(item_content) + "\n")
    
    @classmethod
    def from_config(cls, cfg):
        
        game_config = dict()
        seed = cfg.get("seed", 1234)
        test = cfg.get("test", False)
        game_level = cfg.get("game_level", 1) # The level of babyai challenge
        max_episode_steps = cfg.get("max_episode_steps", 50) # The maximum number of steps per episode
        obs_to_reward = cfg.get("obs_to_reward", None) # The states that will be used to calculate the reward
        difficulty = cfg.get("difficulty", None) # The difficulty of the game, can be "easy","hard"
        game_name = all_levels[game_level]
        
        env = cls(max_episode_steps=max_episode_steps,
                  game_name=game_name,
                  seed=seed,
                  game_config=game_config,
                  obs_to_reward=obs_to_reward,
                  difficulty=difficulty,
        )
        return env
    


all_levels = {
    1: "BabyAI-GoToRedBallGrey-v0",
    2: "BabyAI-GoToRedBall-v0",
    3: "BabyAI-GoToRedBallNoDists-v0",
    4: "BabyAI-GoToObjS6-v0",
    5: "BabyAI-GoToLocalS8N7-v0",
    6: "BabyAI-GoToObjMazeS7-v0",
    7: "BabyAI-GoToImpUnlock-v0",
    8: "BabyAI-GoToSeqS5R2-v0",
    9: "BabyAI-GoToRedBlueBall-v0",
    10: "BabyAI-GoToDoor-v0",
    11: "BabyAI-GoToObjDoor-v0",
    12: "BabyAI-Open-v0",
    13: "BabyAI-OpenRedDoor-v0",
    14: "BabyAI-OpenDoorLoc-v0",
    15: "BabyAI-OpenRedBlueDoorsDebug-v0",
    16: "BabyAI-OpenDoorsOrderN4Debug-v0",
    17: "BabyAI-Pickup-v0",
    18: "BabyAI-UnblockPickup-v0",
    19: "BabyAI-PickupLoc-v0",
    20: "BabyAI-PickupDistDebug-v0",
    21: "BabyAI-PickupAbove-v0",
    22: "BabyAI-PutNextLocalS6N4-v0",
    23: "BabyAI-PutNextS7N4Carrying-v0",
    24: "BabyAI-Unlock-v0",
    25: "BabyAI-UnlockLocalDist-v0",
    26: "BabyAI-KeyInBox-v0",
    27: "BabyAI-UnlockPickupDist-v0",
    28: "BabyAI-BlockedUnlockPickup-v0",
    29: "BabyAI-UnlockToUnlock-v0",
    30: "BabyAI-ActionObjDoor-v0",
    31: "BabyAI-FindObjS7-v0",
    32: "BabyAI-KeyCorridorS6R3-v0",
    33: "BabyAI-OneRoomS20-v0",
    34: "BabyAI-MoveTwoAcrossS8N9-v0",
    35: "BabyAI-SynthS5R2-v0",
    36: "BabyAI-SynthLoc-v0",
    37: "BabyAI-SynthSeq-v0",
    38: "BabyAI-MiniBossLevel-v0",
    39: "BabyAI-BossLevel-v0",
    40: "BabyAI-BossLevelNoUnlock-v0"
}


IDX_TO_ACTION = { 0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}

ACTION_TO_IDX = { "left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5, "done": 6}    

IDX_TO_OBJECT = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}
    
    
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

IDX_TO_STATE = {
    0: "open",
    1: "closed",
    2: "locked",
}

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

IDX_TO_COLOR = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}

DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class BabyAIEnv:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.games = []

    def create(self):
        try:
            idx = self._max_id
            self.info[idx] = {"deleted": False, "done": False}
            self._max_id += 1
            return {"id": idx}
        except Exception as e:
            return {"error": str(e)}

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            self.env[idx].step(action)
            action_space = "\nAvailable actions: ["
            for action in self.env[idx]._get_action_space():
                action_space += "\"" + action + "\", "
            if action_space[-1] != "[":
                action_space = action_space[:-2]
                action_space += "]"
            payload = {
                "observation": self.env[idx]._get_obs() + action_space,
                "reward": self.env[idx].reward,
                "score": self.info[idx]["score"] + self.env[idx].reward,
                "done": self.env[idx]._is_done(),
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def reset(self, idx: int, data_idx: int):
        try:
            self._check_id(idx, True)
            self.env[idx] = BabyAI(game_name=all_levels[data_idx % 40 + 1], seed=data_idx // 40)
            self.env[idx].reset()
            action_space = "\nAvailable actions: ["
            for action in self.env[idx]._get_action_space():
                action_space += "\"" + action + "\", "
            if action_space[-1] != "[":
                action_space = action_space[:-2]
                action_space += "]"
            payload = {
                "observation": "Your goal: " + self.env[idx]._get_goal() + "\n" + self.env[idx]._get_obs() + action_space,
                "reward": self.env[idx].reward,
                "score": self.env[idx].reward,
                "deleted": False,
                "done": self.env[idx]._is_done(),
            }
            self.info[idx].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise ValueError(f"The id {idx} is not valid.")
        if self.info[idx]["deleted"]:
            raise ValueError(f"The task with environment {idx} has been deleted.")
        if not is_reset and self.info[idx]["done"]:
            raise ValueError(f"The task with environment {idx} has finished.")


server = BabyAIEnv()
