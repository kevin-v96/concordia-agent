import collections
from concordia.typing import entity

from concordia.language_model import language_model

def make_prompt(deque: collections.deque[str]) -> str:
    """Makes a string prompt by joining all observations, one per line."""
    return "\n".join(deque)

class SimpleLLMAgent(entity.Entity):
    def __init__(self, model: language_model.LanguageModel):
        self._model = model
        # Container (circular queue) for observations.
        self._memory = collections.deque(maxlen=5)

    @property
    def name(self) -> str:
        return 'Dummy'

    def act(self, action_spec = entity.ActionSpec) -> str:
        prompt = make_prompt(self._memory)
        print(f"*****\nDEBUG: {prompt}\n*****")
        return self._model.sample_text(
            "You are a person.\n"
            f"Your name is {self.name} and your recent observations are:\n"
            f"{prompt}\n What should you do next?"
        )
    
    def observe(self, observation: str) -> None:
        # Push a new observation into the memory, if there are too many, the oldest
        # one will be automatically dropped.
        self._memory.append(observation)

def get_agent(model):
    agent = SimpleLLMAgent(model)
    return agent