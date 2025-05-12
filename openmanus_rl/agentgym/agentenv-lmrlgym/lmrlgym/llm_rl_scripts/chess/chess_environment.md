# How to Use Chess and Endgames Tasks

## Overview of the Tasks

**Chess.** We create a text-based chess task to test the strategic decision-making, credit assignment, and trajectory stitching abilities of an RL algorithm. This tests trajectory stitching, because in the dataset good moves are made in games that resulted in victory, and bad moves are made in games that led to defeat. Therefore this requires trajectory stitching and credit assignment acorss the different game outcomes.

We use FEN (Forsyth-Edwards Notation) notation to represent the board state at each turn and we utilize the SAN (Short Algebraic Notation) to represent each action, both of which are standard notations used by the chess community. To generate the data, we have an agent Stockfish 15.1 of various strengths play against another environment Stockfish engine with elo 1200. The agent receives a reward of 1 for a victorious game, -1 for a loss, 0 for non-terminal actions, and -1 for illegal moves.


**Endgames (Theoretical Chess Endgames)** Chess endgames provide a simpler and more goal-directed variation of the chess task. By focusing on the endgame we emphasize strategy rather than memorizing opening moves. A classic theoretical endgame position consists of a position where the only pieces on the board are the two kings and the queen. Although the board position appears simple, a sequence of carefully calculated moves is required to win. A simpler board state allows language models to make progress without fewer computational resources. All choices we make regarding board state representation and reward function remain the same as for Chess.

## Datasets

**Chess** Use `train_bc.jsonl` to train bc and `train_bc_filtered.jsonl` to train bc and filtered bc respectively. Use `train_trajectories.jsonl` to train ILQL, and MC Returns. The validation data splitting is handled within the training scripts.

**Endgames** Use `train_bc.jsonl` as train dataset and `val_bc.jsonl` as validation datasets and `train_filtered_bc.jsonl`, `val_filtered_bc.jsonl` as train and validation datasets for filtered bc. Use `train.jsonl` and `val.jsonl` for finetuning with RL. 

## How to Run Experiments

### BC

**Full Games**

`python llm_rl_scripts/chess/bc/train_full_games_bc.py HF gpt2 PATH_TO_DATA`

To do filtered BC, set the filtered flag.

**Endgames**

`python llm_rl_scripts/chess/bc/train_endgames_bc.py HF gpt2 YOUR_PATH/train_bc.jsonl YOUR_PATH/val_bc.jsonl YOUR_PATH/test_positions.jsonl`


### ILQL 

`python llm_rl_scripts/chess/train_endgames_ilql.py PARAMS BC_CHECKPOINT_PATH `

## How to Evaluate

**Chess.** To evaluate the full chess model we play 1000 full games against stockfish of elo 1200.

**Endgames.** To evaluate the chess agent in endgame positions, we select 645 positions not contained in the training dataset and which are not trivially solvable. A trivially solvable position is one that Stockfish declares to be simple or a victory in less than 15 moves. We then have the chess agent play one game from each position of these positions and keep these positions fixed for evaluation purposes. In this case we consider filtered BC to be training BC on all of the trajectories which ended in a victory.



