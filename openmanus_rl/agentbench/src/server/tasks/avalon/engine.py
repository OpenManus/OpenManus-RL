from typing import ClassVar, List, Optional, Dict
import numpy as np
from pydantic import BaseModel
from .avalon_exception import AvalonEnvException

class AvalonBasicConfig(BaseModel):
    r"""Avalon game configuration

    Class Variables:
        QUEST_PRESET:
            - Detail: Presets for each quest under various game settings (number of players)
            - Typing: Dict[num_players: List[List[num_good, num_evil], List[num_players_for_quest], List[num_fails_for_quest]]]
        MAX_ROUNDS: Number of the maximum rounds
        PHASES: Map from id to different phases
        ROLES: Map from id to different roles

    Args:
        merlin (bool): Whether Merlin is in the game
        percival (bool): Whether Percival is in the game
        morgana (bool): Whether Morgana is in the game
        mordred (bool): Whether Mordred is in the game
        oberon (bool): Whether Oberon is in the game
    
        num_players (int): Number of players in the game
        num_good (int): Number of good players in the game
        num_evil (int): Number of evil players in the game
        num_players_for_quest (list): Number of players for each quest
        num_fails_for_quest (list): Number of rejects required for the failure of each quest

        preset_flag (bool): Whether the game is initialized with presets

    Method:
        :method:`from_num_players` (@classmethod): instantiate the class from number of players
        :method:`from_presets` (@classmethod): instantiate the class from presets
    """

    QUEST_PRESET: ClassVar =   {5 : [[3,2] , [2,3,2,3,3], [1,1,1,1,1], ] , 
                                6 : [[4,2] , [2,3,4,3,4], [1,1,1,1,1],] , 
                                7 : [[4,3] , [2,3,3,4,4], [1,1,1,2,1],] , 
                                8 : [[5,3] , [3,4,4,5,5], [1,1,1,2,1],] , 
                                9 : [[6,3] , [3,4,4,5,5], [1,1,1,2,1],] , 
                                10 : [[6,4] , [3,4,4,5,5], [1,1,1,2,1],]}
    
    MAX_ROUNDS: ClassVar = 5
    PHASES: ClassVar = {0 : "Team Selection", 1 : "Team Voting", 2 : "Quest Voting", 3 : "Assassination"}
    ROLES: ClassVar = {0 : "Merlin", 1 : "Percival", 2 : "Morgana", 3 : "Mordred", 4 : "Oberon", 5 : "Servant", 6 : "Minion", 7 : "Assassin"}
    ROLES_REVERSE: ClassVar = {v: k for k, v in ROLES.items()}

    merlin: bool = True
    percival: bool = False
    morgana: bool = False
    mordred: bool = False
    oberon: bool = False

    num_players: int
    num_good: int
    num_evil: int
    num_players_for_quest: list
    num_fails_for_quest: list

    preset_flag: bool = False


    @classmethod
    def from_num_players(cls, num_players: int, **kwargs) -> 'AvalonBasicConfig':
        num_evil = cls.QUEST_PRESET[num_players][0][1]
        num_good = num_players - num_evil
        num_players_for_quest = cls.QUEST_PRESET[num_players][1]
        num_fails_for_quest = cls.QUEST_PRESET[num_players][2]

        return cls(
            num_players=num_players,
            num_good=num_good,
            num_evil=num_evil,
            num_players_for_quest=num_players_for_quest,
            num_fails_for_quest=num_fails_for_quest,
            preset_flag=False,
            **kwargs
        )
    
    @classmethod
    def from_presets(cls, presets: Dict) -> 'AvalonBasicConfig':
        num_players = presets['num_players']

        num_evil = cls.QUEST_PRESET[num_players][0][1]
        num_good = num_players - num_evil
        num_players_for_quest = cls.QUEST_PRESET[num_players][1]
        num_fails_for_quest = cls.QUEST_PRESET[num_players][2]

        return cls(
            num_players=num_players,
            num_good=num_good,
            num_evil=num_evil,
            num_players_for_quest=num_players_for_quest,
            num_fails_for_quest=num_fails_for_quest,
            preset_flag=True,
        )

    
class AvalonGameEnvironment():
    r"""Avalon game environment, call methods to access environment.
    
    There are two ways to initialize the environment:
    1. Directly call the constructor with the game presets, and then call the method :method:`reset` to initialize the game
    2. Call the class method :method:`from_presets` to instantiate the environment with game presets, which includes role information
    
    When calling the class method :method:`from_presets`, the following presets are required:
    - role_names (List[str]): List of role names for each player
    - num_players (int): Number of players in the game
    - quest_leader (int): The id of the quest leader
    """
    def __init__(self, config: AvalonBasicConfig) -> None:
        for key, value in config.dict().items():
            setattr(self, key, value)

        self.config = config

        if not self.preset_flag:
            print("New Game!")
            self.reset()

    @classmethod
    def from_num_players(cls, num_players: Dict) -> 'AvalonGameEnvironment':
        r"""Instantiate the environment with number of players"""
        config = AvalonBasicConfig.from_num_players(num_players)
        cls.config = config

        return cls(config)

    @classmethod
    def from_presets(cls, presets: Dict) -> 'AvalonGameEnvironment':
        r"""Instantiate the environment with game presets"""
        config = AvalonBasicConfig.from_presets(presets)
        cls.config = config

        print(presets)

        num_players = presets['num_players']
        quest_leader = presets['quest_leader']
        role_names = presets['role_names']
        role_ids = [config.ROLES_REVERSE[role_name] for role_name in role_names]

        is_good = np.full(num_players, True).tolist()
        for idx, role in enumerate(role_names):
            if role in ["Morgana", "Mordred", "Oberon", "Minion", "Assassin"]:
                is_good[idx] = False

        cls.roles = np.array(role_ids)
        cls.role_names = role_names
        cls.is_good = np.array(is_good)
        cls.quest_leader = quest_leader

        cls.round = 0
        cls.quest = 0
        cls.phase = 0
        cls.turn = 0
        cls.done = False
        cls.good_victory = False

        cls.quest_results = []
        cls.quest_team = []
        cls.team_votes = []
        cls.quest_votes = []
        

        return cls(config)

    def reset(self):
        '''
        Reset game environment
        '''

        # reset game trackers
        self.round = 0
        self.quest = 0
        self.phase = 0
        self.turn = 0
        self.done = False
        self.good_victory = False
        self.quest_leader = np.random.randint(0, self.num_players - 1)

        self.quest_results = []
        self.quest_team = []
        self.team_votes = []
        self.quest_votes = []

        # reassign roles
        return self.assign_roles()
    
    def assign_roles(self):
        '''
        assigns roles to players
        '''
        self.roles = np.full(self.num_players, 5)
        self.is_good = np.full(self.num_players, True)

        # choose num_evil players to be evil
        evil_players = np.random.choice(range(self.num_players), self.num_evil, replace = False)
        self.is_good[evil_players] = False

        # create evil roles
        evil_roles = [7]
        if self.morgana:
            evil_roles.append(2)
        if self.mordred:
            evil_roles.append(3)
        if self.oberon:
            evil_roles.append(4)
        
        # fill rest of evil roles with 6
        evil_roles += [6] * (self.num_evil - len(evil_roles))

        # assign evil roles randomly
        self.roles[evil_players] = np.random.choice(evil_roles, self.num_evil, replace = False)

        # create good roles
        good_roles = []
        if self.merlin:
            good_roles.append(0)
        if self.percival:
            good_roles.append(1)

        # fill rest of good roles with 5
        good_roles += [5] * (self.num_good - len(good_roles))

        # assign good roles randomly
        good_players = np.where(self.is_good)[0]
        self.roles[good_players] = np.random.choice(good_roles, self.num_good, replace = False)

        # return list of role names
        return [self.config.ROLES[role] for role in self.roles]

    def get_role(self, player):
        '''
        returns tuple of role index, role name, and whether player is good
        '''
        return (self.roles[player], self.config.ROLES[self.roles[player]], self.is_good[player])
    
    def get_roles(self):
        '''
        returns list of tuples of role index, role name, and whether player is good
        '''
        return [(role, self.config.ROLES[role], self.is_good[player]) for player, role in enumerate(self.roles)]
    
    def get_partial_sides(self, player):
        '''
        returns list of the sides of other players that player knows
        '''
        
        # if player is Merlin or evil, return all sides
        if self.roles[player] == 0 or not self.is_good[player]:
            return self.is_good
        # otherwise return list of -1 for unknown
        else:
            return [-1 if i != player else 1 for i in range(self.num_players)]
    
    def get_phase(self):
        '''
        returns tuple of phase index and phase name
        '''
        return (self.phase, self.config.PHASES[self.phase])
    
    def get_quest_leader(self):
        '''
        returns quest leader
        '''
        return self.quest_leader
    
    def get_team_size(self):
        '''
        returns team size
        '''
        return self.num_players_for_quest[self.turn]
    
    def choose_quest_team(self, team: frozenset, leader):
        '''
        chooses quest team
        team: list of players on team
        returns: (next phase, whether game is done, next quest leader)
        '''
        # check if game ended or not
        if self.done:
            raise AvalonEnvException("Game ended")

        # check if it is team selection phase. if not, raise error
        if self.phase != 0:
            raise AvalonEnvException("Not in team selection phase")

        # check if team size is valid
        # if np.sum(team) != self.num_players_for_quest[self.round]:
        #     raise AvalonEnvException("Invalid team size")

        if len(team) != self.num_players_for_quest[self.turn]:
            raise AvalonEnvException("Invalid team size")

        # check if leader is quest leader
        if leader != self.quest_leader:
            raise AvalonEnvException("Not quest leader")

        self.quest_team = team

        # move to next phase
        self.phase += 1
        self.quest_leader = (self.quest_leader + 1) % self.num_players

        return (self.phase, self.done, self.quest_leader)
    
    def get_current_quest_team(self):
        '''
        returns list of players on quest team
        '''
        return self.quest_team

    def gather_team_votes(self, votes: List):
        '''
        votes on quest team: list, 0 for reject, 1 for accept
        returns (next phase, whether game is done, whether team is accepted)
        '''
        # check if game ended or not
        if self.done:
            raise AvalonEnvException("Game ended")

        # check if it is team voting phase. if not, raise error
        if self.phase != 1:
            raise AvalonEnvException("Not in team voting phase")

        # check if votes is valid
        if len(votes) != self.num_players:
            raise AvalonEnvException("Invalid number of votes")

        self.team_votes = votes

        # if this is the MAX_ROUNDS round, then team automatically passes
        if self.round == self.config.MAX_ROUNDS -1:
            self.phase += 1
            self.round = 0
            return (self.phase, self.done, True)

        # if strict majority accepts, move to next phase. Otherwise back to team selection
        if sum(votes) > self.num_players / 2:
            self.phase += 1
            self.round = 0
            return (self.phase, self.done, True)
        else:
            self.phase = 0
            self.round += 1
            return (self.phase, self.done, False)
        
    def get_quest_team_votes(self):
        '''
        returns list of votes on quest team
        '''
        return self.quest_team_votes
    
    def gather_quest_votes(self, votes: List):
        '''
        votes on quest: list, 0 for fail, 1 for pass
        returns: (next phase, whether game is done, whether the quest succeeded, number of fails)
        '''
        # check if game ended or not
        if self.done:
            raise AvalonEnvException("Game ended")

        # check if it is quest voting phase. if not, raise error
        if self.phase != 2:
            raise AvalonEnvException("Not in quest voting phase")

        # check if votes is valid
        if len(votes) != self.num_players_for_quest[self.turn]:
            raise AvalonEnvException("Invalid number of votes")

        self.quest_votes = votes

        num_fails = len(votes) - sum(votes)

        # if number of fails is greater to or equal to number of fails allowed, quest fails
        if (num_fails) >= self.num_fails_for_quest[self.turn]:
            
            self.quest_results.append(False)
            self.turn += 1

            # end game if 3 quests failed. Otherwise go to team selection phase
            if len(self.quest_results) - sum(self.quest_results) == 3:
                self.done = True
                self.good_victory = False
                return (self.phase, self.done, False, num_fails)
            else:
                self.phase = 0
                return (self.phase, self.done, False, num_fails)

        else:
            self.quest_results.append(True)
            self.turn += 1

            # go to assassination phase if 3 quests succeeded. Otherwise go to team selection phase
            if sum(self.quest_results) == 3:
                self.phase += 1
            else:
                self.phase = 0
            return (self.phase, self.done, True, num_fails)
        
    def get_assassin(self):
        '''
        returns assassin
        '''
        return np.where(self.roles == 7)[0][0]
        
    def choose_assassination_target(self, player, target):
        '''
        chooses assassination target
        returns: (next phase, whether game is done, whether good wins)
        game ends
        '''
        # check if game ended or not
        if self.done:
            raise AvalonEnvException("Game ended")

        # check if it is assassination phase. if not, raise error
        if self.phase != 3:
            raise AvalonEnvException("Not in assassination phase")

        # check if player is assassin
        if self.roles[player] != 7:
            raise AvalonEnvException("Not assassin")

        # check if player is good
        if self.is_good[player]:
            raise AvalonEnvException("Assassin cannot be good")

        self.done = True

        # check if target is merlin
        if self.roles[target] == 0:
            self.good_victory = False
            return (self.phase, self.done, False)
        
        # check if at least 3 successfuel quests
        if sum(self.quest_results) >= 3:
            self.good_victory = True
            return (self.phase, self.done, True)
        
        # otherwise evil wins
        self.good_victory = False
        return (self.phase, self.done, False)

if __name__ == "__main__":
    config = AvalonBasicConfig.from_num_players(5)
    env = AvalonGameEnvironment.from_presets({
        'num_players': 5,
        'quest_leader': 0,
        'role_names': ['Servant', 'Percival', 'Morgana', 'Mordred', 'Oberon']
    })
    
    print(env.get_role(0))
    
    env = AvalonGameEnvironment.from_num_players(5)

    print(env.get_role(0))
    # print(config.dict())
    # print(env.roles)
    # print(env.is_good)
    # print(config.ROLES)
    # print(config.ROLES_REVERSE)