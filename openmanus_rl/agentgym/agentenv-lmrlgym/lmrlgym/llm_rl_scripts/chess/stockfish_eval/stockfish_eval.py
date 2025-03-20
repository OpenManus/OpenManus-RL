from stockfish import Stockfish
from llm_rl_scripts.chess.env.data import get_data_from_bucket
from llm_rl_scripts.chess.env.env import ChessEnv 
import os
from LLM_RL.utils import convert_path
from IPython import embed
import chess
from JaxSeq.utils import create_path
import json
from tqdm.auto import tqdm

CHESS_ENGINE_PATH = os.environ.get("CHESS_ENGINE_PATH", convert_path("stockfish/stockfish-ubuntu-20.04-x86-64-avx2"))
OUTPUTS_PATH = "data/outputs/chess/stockfish_eval/"
def get_stockfish_eval(env: ChessEnv) -> float:
    done = False 
    rewards = []
    params = {"Threads": 1, "UCI_Elo": 1200}
    stockfish = Stockfish(CHESS_ENGINE_PATH, parameters=params)
    while not done:
        stockfish.set_fen_position(env.board.fen())
        board = chess.Board(env.board.fen())
        move : str = stockfish.get_best_move()
        try:
            ch_move : chess.Move = chess.Move.from_uci(move)
        except:
            print(move)
        san_move = board.san(ch_move)
        st, rew, done, _ = env.step(san_move)

        rewards.append(rew)
    return sum(rewards)

def do_stockfish_evals(num_evals, starting_position):
    env = ChessEnv(from_position=starting_position)
    scores = []
    for _ in range(num_evals):
        env.reset()
        score = get_stockfish_eval(env)
        scores.append(score)
    return scores

def full_game_eval():
    starting_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'
    num_evals = 50
    scores = do_stockfish_evals(num_evals, starting_position)
    print(scores)
    print(sum(scores) / len(scores))
    score_json = {"avg_reward": sum(scores) / len(scores)}
    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'stockfish_eval.json'), 'w') as f:
        json.dump(score_json, f)

def endgames_eval():
    bucket_name = "rl-llm-bench-dataset-internal"
    blob_name = "endgames/test_positions.jsonl"
    test_positions = get_data_from_bucket(bucket_name, blob_name)
    test_positions = [position.replace("\n", "").replace("\"", "") for position in test_positions if position != ""]
    scores = []
    for position in tqdm(test_positions):
        score = do_stockfish_evals(1, position)[0]
        scores.append(score)
    score_json = {"avg_reward": sum(scores) / len(scores)}
    return score_json
        
def main():
    results = endgames_eval()
    create_path(OUTPUTS_PATH)
    with open(os.path.join(OUTPUTS_PATH, 'stockfish_eval.json'), 'w') as f:
        json.dump(results, f)
    
    
if __name__ == "__main__":
    main()
    