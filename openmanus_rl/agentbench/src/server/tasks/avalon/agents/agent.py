from typing import List
from ..engine import AvalonBasicConfig
class Agent:
    r"""The base class for all agents.

    Args:
        id (int): The Player id of the agent.
        role (int): The role (id) of the agent.
        config (AvalonBasicConfig): The config of the agent.

    To implement your own agent, subclass this class and implement the following methods:
        - :method:`Agent.propose_team`
        - :method:`Agent.vote_on_team`
        - :method:`Agent.vote_on_mission`
    """
    def __init__(self, id: int, role: int, config: AvalonBasicConfig) -> None:
        self.id = id
        self.name = f"Player {id}"
        self.role = role
        self.role_name = config.ROLES[role]
        self.config = config


    def propose_team(self, mission_id: int) -> frozenset[int]:
        r"""Propose a team of given size.

        Args:
            mission_id (int): The id of the mission. team_size = config.num_players_for_quest[mission_id]

        Returns:
            frozenset[int]: The list of player ids to be included in the team.
        """
        raise NotImplementedError
    

    def vote_on_team(self, mission_id: int, team: frozenset[int]) -> bool:
        r"""Vote on a given team.

        Args:
            mission_id (int): The id of the mission. num_fails = self.config.num_fails_for_quest[mission_id]
            team (frozenset[int]): The list of player ids included in the team.

        Returns:
            bool: The vote result.
        """
        raise NotImplementedError
    
    def vote_on_mission(self, mission_id: int, quest_team: frozenset[int]) -> bool:
        r"""Vote on a quest (team).

        Args:
            mission_id (int): The id of the mission. num_fails = self.config.num_fails_for_quest[mission_id]
            quest_team (frozenset[int]): The list of player ids included in the quest.
        
        Returns:
            bool: The vote result.
        """
        raise NotImplementedError
    
    
    def assassinate(self, num_players: int) -> int:
        r"""Assassinate a player.

        Args:
            num_players (int): The number of players in the game.

        Returns:
            int: The id of the player to assassinate. The id is in the range [0, num_players).
        """
        raise NotImplementedError
    

    def get_believed_sides(self, num_players: int) -> List[float]:
        r"""Get the believed sides of all players.

        Args:
            num_players (int): The number of players in the game.
        
        Returns:
            List[float]: The list of believed sides (probability) of all players.
        """
        raise NotImplementedError