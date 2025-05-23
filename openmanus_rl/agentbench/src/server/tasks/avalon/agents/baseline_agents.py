import random
from ..engine import AvalonBasicConfig, AvalonGameEnvironment
import itertools
import warnings
from typing import List, Dict, Tuple
from .agent import Agent

class NaiveAgent(Agent):
    r"""Agent with naive strategies.

    When creating an agent, the following information is needed:
    - id (int): The Player id of the agent.
    - role (int): The role (id) of the agent.
    - config (AvalonBasicConfig): The config of the basic game info.
    
    Please refer to the paper https://arxiv.org/pdf/2310.05036.pdf for details.
    """

    def __init__(self, id: int, role: int, config: AvalonBasicConfig, name: str, side: int=None, sides: List[int] = None, **kwargs):
        super().__init__(id, role, config)
        self.name = name
        self.id = id
        self.config = config
        self.role = role
        self.team = None
        self.side = side # 1 for good, 0 for evil
        self.history = None
        if sides is None:
            self.player_sides = [-1] * self.config.num_players # -1 for unknown, 0 for evil, 1 for good
            self.player_sides[id] = side

            # if role is 0 (Merlin) or side is evil, warn that player_sides are not seen
            if role == 0 or side == 0:
                warnings.warn("Merlin and evil players did not see player sides in initialization")
        else:
            self.player_sides = sides
        
        random.seed(0, version=1)
        

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
    
    async def propose_team(self, mission_id: int, **kwargs):
        return frozenset(random.sample(range(0, self.config.num_players), self.config.num_players_for_quest[mission_id]))
    
    async def vote_on_team(self, mission_id, team: frozenset, **kwargs):
        return random.choice([0, 1])
    
    async def vote_on_mission(self, mission_id, team: frozenset, **kwargs):
        return self.side
    
    async def initialize_game_info(self, player_list):
        pass

    def assign_side(self, side):
        self.side = side

    def assign_role(self, role):
        self.role = role

    def see_sides(self, sides):
        self.player_sides = sides

    async def assignHistory(self, history):
        self.history = history

    async def observe_mission(self, team: frozenset, mission_id: int, **kargs):
        pass

    async def observe_team_result(self, mission_id, team: frozenset, votes: List[int], outcome: Tuple):
        pass

    async def assassinate(self):
        return random.randint(0, self.config.num_players-1)
    
    async def get_believed_sides(self, **kwargs):
        return [0.5 if side == -1 else side for side in self.player_sides]
    

class NaiveMinion(NaiveAgent):
    
    def __init__(self, id: int, name: str, config: AvalonBasicConfig, side: int=0, role_name: str="Minion", role: int=6, sides: List[int]=None, **configs):
        assert role == 6
        super().__init__(
            id      =   id,
            name    =   name,
            config  =   config,
            side    =   side,
            role    =   role,
            sides   =   sides
            )

    async def vote_on_mission(self, mission_id: int, team: frozenset, **kwargs):
        num_fails = self.config.num_fails_for_quest[mission_id]

        # if less than num_fails evil players on the team, vote success
        if sum([self.player_sides[i] == 0 for i in team]) < num_fails:
            return 1
        # if equal to num_fails evil players on the team, vote fail
        elif sum([self.player_sides[i] == 0 for i in team]) == num_fails:
            return 0
        # if there are only two evil players in the game (ie. the assassin is on the team) and num_fails == 1, vote success. otherwise, vote fail
        elif self.config.num_evil == 2 and num_fails == 1:
            return 1
        else:
            return 0
        
    async def vote_on_team(self, mission_id: int, team: frozenset, **kwargs):
        # approve if there are at least x evil player(s) on the team, where x is number of fails required for this mission
        num_fails = self.config.num_fails_for_quest[mission_id]
        if sum([self.player_sides[i] == 0 for i in team]) >= num_fails:
            return 1
        else:
            return 0
        
    async def propose_team(self, mission_id: int, **kwargs):
        num_fails = self.config.num_fails_for_quest[mission_id]
        # choose evil team with x-1 other evil player(s), where x is number of fails required for this mission, plus the minion
        evil_team = random.sample([i for i in range(self.config.num_players) if self.player_sides[i] == 0 and i != self.id], num_fails - 1) + [self.id]

        # propose a random team that includes evil_team and y-x good player(s), where y is number of players required for this mission
        return frozenset(random.sample([i for i in range(self.config.num_players) if i not in evil_team and self.player_sides[i] == 1], self.config.num_players_for_quest[mission_id] - num_fails) + evil_team)
        
class NaiveAssassin(NaiveAgent):
    
    def __init__(self, id: int, name: str, config: AvalonBasicConfig, side: int=0, role_name: str="Assassin", role: int=7, sides: List[int]=None, **configs):
        assert role == 7
        super().__init__(
            id      =   id,
            name    =   name,
            config  =   config,
            side    =   side,
            role    =   role,
            sides   =   sides
            )

    async def vote_on_mission(self, mission_id: int, team: frozenset, **kwargs):
        num_fails = self.config.num_fails_for_quest[mission_id]

        # if less than num_fails evil players on the team, vote success
        if sum([self.player_sides[i] == 0 for i in team]) < num_fails:
            return 1
        # else vote fail
        else:
            return 0
        
    async def vote_on_team(self, team: frozenset, mission_id: int, **kwargs):
        # approve if there are at least x evil player(s) on the team, where x is number of fails required for this mission
        num_fails = self.config.num_fails_for_quest[mission_id]
        if sum([self.player_sides[i] == 0 for i in team]) >= num_fails:
            return 1
        else:
            return 0
        
    async def propose_team(self, mission_id: int, **kwargs):
        num_fails = self.config.num_fails_for_quest[mission_id]
        # choose evil team with x-1 other evil player(s), where x is number of fails required for this mission, plus the assassin
        evil_team = random.sample([i for i in range(self.config.num_players) if self.player_sides[i] == 0 and i != self.id], num_fails - 1) + [self.id]

        # propose a random team that includes evil_team and y-x good player(s), where y is number of players required for this mission
        return frozenset(random.sample([i for i in range(self.config.num_players) if i not in evil_team and self.player_sides[i] == 1], self.config.num_players_for_quest[mission_id] - num_fails) + evil_team)
        
    async def assassinate(self, **kwargs):
        # assassinate a random good player
        return random.choice([i for i in range(self.config.num_players) if self.player_sides[i] == 1])

class NaiveMerlin(NaiveAgent):

    def __init__(self, id: int, name: str, config: AvalonBasicConfig, side: int=1, role: int=0, sides: List[int]=None, **configs):
        assert role == 0
        super().__init__(
            id      =   id,
            name    =   name,
            config  =   config,
            side    =   side,
            role    =   role,
            sides   =   sides
            )
    
    async def vote_on_team(self, team: frozenset, mission_id: int, **kwargs):
        # approve if there are no evil players on the team
        if any([self.player_sides[i] == 0 for i in team]):
            return 0
        else:
            return 1
        
    async def propose_team(self, mission_id: int, **kwargs):
        # propose a random team with all good players that includes Merlin
        return frozenset(random.sample([i for i in range(self.config.num_players) if self.player_sides[i] != 0 and i != self.id], self.config.num_players_for_quest[mission_id] - 1) + [self.id])

class NaiveServant(NaiveAgent):

    def __init__(self, id: int, name: str, config: AvalonBasicConfig, side: int=1, role: int=5, sides: List[int]=None, lexigraphic: bool=True, **configs):
        assert role == 5
        super().__init__(
            id      =   id,
            name    =   name,
            config  =   config,
            side    =   side,
            role    =   role,
            sides   =   sides
            )

        # maintain a list of all possible combinations of player sides
        self.possible_player_sides = self.generate_possible_player_sides(self.player_sides, self.config.num_evil)
        self.player_side_probabilities = [1/len(self.possible_player_sides)] * len(self.possible_player_sides)

        # generate team preferences for first mission
        self.team_preferences = self.generate_team_preferences(0)

        # record the largest and most recent team that succeeded
        self.largest_successful_team = None

        self.lexigraphic = lexigraphic

    def generate_possible_player_sides(self, sides: List[int], num_evils: int):
        '''Util func of NaiveServant
        
        generates a list of all possible combinations of player sides given a list of known sides and unknown sides recursively as well number of unknown evils
        '''
        out = []
        # if there are no unknown sides, return the list of sides   
        if -1 not in sides:
            return [sides]
        else:
            # find the first unknown side
            unknown_index = sides.index(-1)
            unknown_count = sum([1 for side in sides if side == -1])
            num_good = unknown_count - num_evils
            # recurse on the two possible sides
            for side in [0, 1]:
                if side == 0 and num_evils == 0:
                    continue
                if side == 1 and num_good == 0:
                    continue
                sides_copy = sides.copy()
                sides_copy[unknown_index] = side
                if side == 0:
                    out.extend(self.generate_possible_player_sides(sides_copy, num_evils - 1))
                else:
                    out.extend(self.generate_possible_player_sides(sides_copy, num_evils))    
            return out
        
    def generate_team_preferences(self, mission_id: int):
        '''
        generates preferences across mission teams specified by mission_id
        '''
        team_size = self.config.num_players_for_quest[mission_id]
        # generate list of all possible teams of size team_size, where team should be a set of player ids
        teams = [frozenset(team) for team in itertools.combinations(range(self.config.num_players), team_size)]

        # maintain a list of preferences for each team, and set all preferences to 0
        team_preferences = [0] * len(teams)

        # iterate over possible player sides and side probabilities
        for sides, prob in zip(self.possible_player_sides, self.player_side_probabilities):
            # iterate over possible teams
            for team in teams:
                # if team is all good in sides, increment team preference by prob of side
                if all([sides[i] == 1 for i in team]):
                    team_preferences[teams.index(team)] += prob
        
        # return dictionary mapping teams to preferences
        return dict(zip(teams, team_preferences))
    
    def find_most_prefered_teams(self, team_to_preferences: Dict):
        '''
        returns a list of the most preferred teams, where team_to_preferences is a dictionary mapping teams to preferences
        '''
        # find the maximum preference
        max_preference = max(team_to_preferences.values())
        # return a list of all teams with maximum preference
        max_teams = [frozenset(team) for team, preference in team_to_preferences.items() if preference == max_preference]
        # if there is only one team with maximum preference, return it
        if len(max_teams) == 1:
            return max_teams
        # else return list of teams of max_teams that are subsets of self.largest_successful_team if it is not None and non-empty, otherwise return max_teams
        else:
            if self.largest_successful_team is not None and self.lexigraphic:
                out = [frozenset(team) for team in max_teams if team.issubset(self.largest_successful_team)]
                # print('subset', out)
                if len(out) > 0:
                    return out
                else: # return list of teams of max_teams that are supersets of self.largest_successful_team if it is not None and non-empty, otherwise return max_teams
                    out = [frozenset(team) for team in max_teams if self.largest_successful_team.issubset(team)]
                    # print('superset', out)
                    if len(out) > 0:
                        return out
                    else:
                        return max_teams
            return list(set(max_teams))
    
    async def vote_on_team(self, team: frozenset, mission_id: int, **kwargs):
        # print('vote', self.team_preferences)
        # if team is in most preferred teams, approve, otherwise reject
        return 1 if team in self.find_most_prefered_teams(self.team_preferences) else 0
    
    async def propose_team(self, mission_id: int, **kwargs):
        # propose random team in most preferred teams
        # print('propose', self.team_preferences)
        return random.choice(self.find_most_prefered_teams(self.team_preferences))
    
    async def observe_mission(self, team: frozenset, mission_id: int, num_fails: int, **kargs):
        # if mission succeeded, update largest_successful_team
        if num_fails == 0:
            if self.largest_successful_team is None or len(team) > len(self.largest_successful_team):
                self.largest_successful_team = team

        # set the probability of all sides that have less than num_fails evil players on the team to 0
        for sides, prob in zip(self.possible_player_sides, self.player_side_probabilities):
            if sum([sides[i] == 0 for i in team]) < num_fails:
                self.player_side_probabilities[self.possible_player_sides.index(sides)] = 0

        # normalize probabilities TODO: threre could be situations where sum(self.player_side_probabilities) = 0; should take a look
        try:
            self.player_side_probabilities = [prob / sum(self.player_side_probabilities) for prob in self.player_side_probabilities]
        except:
            self.player_side_probabilities = [0.25 for _ in range(len(self.player_side_probabilities))]
        # print('side probs', self.player_side_probabilities)
        # print(self.possible_player_sides)

        # generate team preferences for next mission, if there is one
        if mission_id < len(self.config.num_players_for_quest)-1:
            self.team_preferences = self.generate_team_preferences(mission_id+1)
        pass

    async def get_believed_sides(self, **kwargs):
        '''
        Return marginal distribution of each player being good based on self.player_side_probabilities and self.possible_player_sides
        '''
        # initialize marginal distribution to 0
        marginal_distribution = [0] * self.config.num_players
        # iterate over possible player sides and side probabilities
        for sides, prob in zip(self.possible_player_sides, self.player_side_probabilities):
            # iterate over players
            for i in range(self.config.num_players):
                # if player is good in sides, increment marginal distribution by prob of side
                if sides[i] == 1:
                    marginal_distribution[i] += prob
        return marginal_distribution
    
NAIVEAGENT_FINDER = {
    'Servant': NaiveServant,
    'Merlin': NaiveMerlin,
    'Assassin': NaiveAssassin,
    'Minion': NaiveMinion
}

def find_naive_agent(**configs):
    return NAIVEAGENT_FINDER[configs["role_name"]](
        name        =   configs["name"],
        num_players =   configs["num_players"],
        session     =   configs["session"],
        role        =   configs["role"],
        role_name   =   configs["role_name"],
        side        =   configs["side"],
        id          =   configs["id"],
        config      =   configs["config"],
        merlin      =   configs["merlin"],
        percival    =   configs["percival"],
        morgana     =   configs["morgana"],
        mordred     =   configs["mordred"],
        oberon      =   configs["oberon"],
        num_good    =   configs["num_good"],
        num_evil    =   configs["num_evil"],
        seed        =   configs["seed"]
    )

if __name__ == "__main__":
    env = AvalonGameEnvironment.from_presets({
        'num_players': 5,
        'quest_leader': 0,
        'role_names': ['Servant', 'Minion', 'Merlin', 'Assassin', 'Servant']
    })
    config = env.config
    player_0 = find_naive_agent['Merlin'](0, 'player_0', config)
        
    

        
        

        


    


    
