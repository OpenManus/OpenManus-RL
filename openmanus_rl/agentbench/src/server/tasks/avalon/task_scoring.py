from .engine import AvalonBasicConfig
import numpy as np

class AvalonScoring():

    def __init__(self, config: AvalonBasicConfig) -> None:
        self.config = config # AvalonConfig object

    def deduction_acc(self, true_player_sides, believed_player_sides) -> float:
        true_player_sides = np.array(true_player_sides)
        # believed_player_sides = np.where(np.array(believed_player_sides) == 0.5, -1, np.array(believed_player_sides))
        believed_player_sides = np.where(np.array(believed_player_sides) >= 0.5, 1, np.array(believed_player_sides))
        believed_player_sides = np.where(np.array(believed_player_sides) < 0.5, 0, np.array(believed_player_sides))

        return np.mean(np.sum(believed_player_sides == true_player_sides, axis=1) / 5)

    def score_deduction(self, true_player_sides, believed_player_sides):
        '''
        N: number of players
        T: number of games
        true_player_sides: T x N matrix of true player sides (0 for evil, 1 for good)
        believed_player_sides: T x N matrix of believed player sides (number in interval [0,1] where 0 is evil and 1 is good)
        scores how well the believed player sides match the true player sides using cross entropy
        returns average cross entropy over all games
        '''
        true_player_sides = np.array(true_player_sides)
        believed_player_sides = np.where(np.array(believed_player_sides) == -1, 0.5, np.array(believed_player_sides))
        believed_player_sides = np.where(np.array(believed_player_sides) >= 1, 0.9999, np.array(believed_player_sides))
        believed_player_sides = np.where(np.array(believed_player_sides) <= 0, 0.0001, np.array(believed_player_sides))
        # believed_player_sides = np.array(believed_player_sides) + 0.0001
        # print(believed_player_sides)

        # believed_player_sides = believed_player_sides / np.sum(believed_player_sides, axis=-1, keepdims=True) * np.sum(true_player_sides, axis=-1, keepdims=True)
        # true_player_sides = true_player_sides / np.sum(true_player_sides, axis=-1, keepdims=True)
        

        # print(believed_player_sides)
        # print(1 - believed_player_sides)
        # print(true_player_sides)
        
        return np.mean(-np.sum(true_player_sides * np.log(believed_player_sides) + (1 - true_player_sides) * np.log(1 - believed_player_sides), axis=1))

    def score_deception(self, other_player_sides, other_player_beliefs):
        '''
        N: number of players
        T: number of games
        other_player_sides: T x (N-1) matrix of other player sides (0 for evil, 1 for good) corresponding to other_player_beliefs
        other_player_beliefs: T x (N-1) matrix of other player beliefs about this player (number in interval [0,1] where 0 is evil and 1 is good)
        scores how good other good players judge this player, taking mean over all games 
        '''

        # how good other good players judge this player
        good_judgement = np.sum(other_player_sides * other_player_beliefs, axis=1) / np.sum(other_player_sides, axis=1)
        return np.mean(good_judgement)
        # how good other evil players judge this player
        # evil_error = np.abs(player_side - np.sum((1 - other_player_sides) * other_player_beliefs, axis=1) / np.sum(1 - other_player_sides, axis=1))
        
        # return np.mean(np.abs(player_side - np.sum(other_player_sides * other_player_beliefs, axis=1) / np.sum(other_player_sides, axis=1)))
        # scores how well the player deceives players not on their side, while not deceiving players on their side, using absolute difference returns average absolute difference over all games and players

    def score_influence_per_game(self, true_vote, vote_outcome):
        '''
        T: number of rounds in game
        true_vote: T x 1 matrix of true vote for team on turn t (0 for reject, 1 for approve)
        vote_outcome: T x 1 matrix of vote outcome for team on turn t (0 for rejected, 1 for approved)
        scores what percentage of time the vote outcome matches the true vote
        '''
        return np.mean(true_vote == vote_outcome)
    
    def score_leadership_per_game(self, vote_outcome):
        '''
        T: number of rounds in game when player is leader
        vote_outcome: T x 1 matrix of vote outcome for team on turn t (0 for rejected, 1 for approved)
        scores what percentage of time the proposed team was approved when the player proposed it
        '''
        return np.mean(vote_outcome)
    
if __name__ == "__main__":
    config = AvalonBasicConfig(num_players=5)
    scoring = AvalonScoring(config)

    true_sides = [[0, 0, 1, 1, 1]]
    believed_sides = [[0.8, 0.3, 0.7, 0.6, 0.5]]

    # believed_sides = np.array(believed_sides) + 0.01
    # # print(np.sum(believed_sides, axis=-1, keepdims=True).reshape(2, 5))
    # believed_sides = believed_sides / np.sum(believed_sides, axis=-1, keepdims=True)
    # print(believed_sides)

    print(scoring.deduction_acc(true_player_sides=true_sides, believed_player_sides=believed_sides))