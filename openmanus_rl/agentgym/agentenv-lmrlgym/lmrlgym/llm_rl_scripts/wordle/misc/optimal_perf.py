from llm_rl_scripts.wordle.env.game import Vocabulary
from llm_rl_scripts.wordle.env.utils import Cache
from llm_rl_scripts.wordle.env.scripted_policies import OptimalPolicy
from LLM_RL.utils import convert_path
from llm_rl_scripts.wordle.env.env import WordleEnvironment
from LLM_RL.environment import interact_environment
import pickle as pkl

if __name__ == '__main__':
    vocab_path = 'llm_rl_scripts/wordle/vocab/wordle_official_400.txt'
    vocab = Vocabulary.from_file(vocab_file=convert_path(vocab_path), fill_cache=False, rng=None)
    policy = OptimalPolicy(vocab=vocab, progress_bar=True)
    with open(convert_path('test_cache.pkl'), 'rb') as f:
        cache_init = pkl.load(f)
    policy.cache = Cache(cache_init)
    env = WordleEnvironment(vocab, require_words_in_vocab=True)

    avg_reward = 0.0
    for i in range(1000):
        transitions = interact_environment(
            env, 
            policy, 
            env_seed=None,
        )[0]
        
        history = transitions[-1].post_transition_history
        rewards = sum([[transition.reward, 0.0] for transition in transitions], [])
        done = transitions[-1].done

        avg_reward = (avg_reward * (i) + sum(rewards)) / (i+1)
        print(avg_reward)

        if i % 100 == 0:
            policy.cache.dump(convert_path('test_cache.pkl'))


