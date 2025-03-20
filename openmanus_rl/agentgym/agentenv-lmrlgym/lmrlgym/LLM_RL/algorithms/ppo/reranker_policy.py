import numpy as np
from LLM_RL.environment import TextPolicy, TextHistory
from typing import Callable, List

class ReRankerSamplePolicy(TextPolicy):
    
    def __init__(self, proposal_fn, score_fn: Callable[[List[TextHistory]], List[float]]):
        self.proposal_fn = proposal_fn
        self.score_fn = score_fn
    
    def act(self, text_history: TextHistory) -> TextHistory:
        proposals = self.proposal_fn(text_history)
        scores = np.asarray(self.score_fn(proposals), dtype=np.float32)
        # sample from scores
        scores = np.exp(scores) / np.exp(scores).sum()
        selected = np.random.choice(len(scores), p=scores)
        # # zip proposals and scores together
        # proposals_and_scores = list(zip(proposals, scores))
        # print(proposals_and_scores)
        return proposals[selected]
    
class ReRankerPolicy(TextPolicy):
    
    def __init__(self, proposal_fn: Callable[[TextHistory], List[TextHistory]], score_fn: Callable[[List[TextHistory]], List[float]]):
        self.proposal_fn = proposal_fn
        self.score_fn = score_fn

    def act(self, text_history: TextHistory) -> TextHistory:
        proposals = self.proposal_fn(text_history)
        scores = self.score_fn(proposals)

        return proposals[np.argmax(np.asarray(scores, dtype=np.float32)).item()]


