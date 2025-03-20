# LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models

This is the official repository for LMRL Gym. You can access the dataset [here](https://rail.eecs.berkeley.edu/datasets/rl-llm-bench-dataset/). 

<!-- ## Baseline Descriptions 

Implementation files and details can be found in `LLM_RL/algorithms`

**MC Returns.** We use Monte-Carlo returns to train a value function for the dataset. The agents then acts according to this value function. We implement this by fine-tuning an LM to predict the reward-to-go of the data at each token, using an action-value function head. This is then used to perturb the logits of the original BC model. -->

## Task Descriptions 

We present 8 tasks aimed at benchmarking RL tasks with language models. 

**Maze**  A maze with a fixed layout and fixed goal and we create two different representations of the state, one that is *partially observed* and one that is *fully observed*. The fully observed representation includes the coordinates of the agent in the maze and the partially observed representation is the history of actions so far in the maze.

**Text-based Navigation (Text-Nav).**
We design a text-based game based on navigation in a house environment using a modified version of the TextWorld engine \citep{textworld}. 
Like in the maze task, we consider a *fully observed* and *partially observed* instantiation of the task. In the former, at each timestep, the full natural language description is provided to the agent, but in the latter, the first two components are omitted.

**Wordle.** In the game wordle the agent is given at most 6 attempts to guess a hidden 5 letter word. After each guess, the agent is told whether each letter in the guessed word is: 1) in the hidden word and in the right position, 2) in the hidden word but not in the right position, or 3) not in the hidden word. 

**Chess.** We create a text-based chess task in order to test the complex decision making, credit assignment, and trajectory stitching properties. We use FEN (Forsyth-Edwards Notation) notation to represent the board state at each turn and we utilize the SAN (Short Algebraic Notation) to represent each action, both of which are standard notations used by the chess community. To generate the data, we have an agent Stockfish 15.1 of various different strengths play against another environment Stockfish engine with elo 1200. The agent receives a reward of 1 for a victorious game, -1 for a loss, 0 for non-terminal actions, and -1 for illegal moves.

**Endgames (Theoretical Chess Endgames).** Chess endgames provide a simpler and more goal-directed variation of the chess task. A classic theoretical endgame position consists of a position where the only pieces on the board are the two kings and the queen. Although the board position appears simple, a sequence of carefully calculated moves is required to win. All choices we make regarding board state representation and reward function remain the same as previously for Chess.

**20Qs (Twenty Questions).**
This task specifically tests information gathering to see if a policy can successfully reason about an unknown subject based on context to determine what it is. In twenty questions, one player (the oracle) thinks of an object, and the other player (the guesser) tries to guess what it is by asking a series of yes-or-no questions and the game continues until the guesser either guesses the correct answer or runs out of questions. We assign the reward as -1 for each guess that is incorrect and 0 for the correct answer.

**Guess (Guess My City).**
This task simulates a more complicated guessing game, where one player (the oracle) is from a specific city, and the other player (the guesser) tries to guess what city the oracle is from. Here, the guesser can ask not only yes and no questions, but can also ask open-ended questions. We assign the reward the same as in the Twenty Questions task.

**Car Dealer**
This task simulates a conversation between a car buyer and a car dealer, each with different strategies on how to get the best deal for themselves, The buyer wants to buy a certain type of car within a certain budget, and the car dealer wants to complete the sale ideally with a high sale price. The reward for the seller is the price of the final car purchase.



## How to Use the Tasks

Files to run each of the tasks can be found in llm_rl_scripts. For each task you will need to train BC before finetuning with RL. For example, to run BC for the Maze task, you would launch the following command: 

``` shell
python llm_rl_scripts/maze/bc/fully_observed_bc.py HF gpt2 PATH_TO_YOUR_DATA --outputs-path bc_checkpoint_path
```

Then convert the BC checkpoint to PARAMS format using 

``` shell
python -m examples_jaxseq.misc.export_checkpoint bc_checkpoint_path
```

You can evaluate your BC checkpoint as follows 

``` shell 
python llm_rl_scripts/maze/bc/eval_bc.py PARAMS bc_checkpoint_path 
```

Then to finetune with ILQL using this checkpoint you run 

``` shell
python llm_rl_scripts/maze/ilql/train_ilql.py PARAMS bc_checkpoint_path PATH_TO_YOUR_DATA --outputs-path ilql_checkpoint_path
```

Finally, to evaluate the results you run 
``` shell
python llm_rl_scripts/maze/ilql/eval_ilql.py PARAMS bc_checkpoint_path PARAMS ilql_checkpoint_path
```

In the subfolder for each task, you can find a README detailing how to run each of the baseline experiments with the baseline hyperparameters. Note that to evaluate PPO policy you can also use the BC evaluation scripts. 


## Installation

### **1. Pull from GitHub**

``` bash
git clone https://github.com/abdulhaim/LMRL-Gym
cd LMRL-Gym
```

### **2. Install dependencies**

Install with conda (cpu, tpu, or gpu).

**Install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip
python -m pip install -e .
```

**Install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip
conda install 'jaxlib=*=*cuda*' jax cuda-nvcc -c conda-forge -c nvidia
python -m pip install -e .
```

**Install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate LLM_RL
python -m pip install --upgrade pip
python -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

### **3. Install JaxSEQ**
``` shell
# navigate to a different directory
cd ~/
git clone https://github.com/Sea-Snell/JaxSEQ
cd JaxSEQ
python -m pip install -e .
```
