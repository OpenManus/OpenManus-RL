from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Agent:

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
