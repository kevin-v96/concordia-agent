from concordia.associative_memory import associative_memory
from concordia.typing import entity
from concordia.language_model import language_model

def make_prompt_associative_memory(memory: associative_memory.AssociativeMemory):
    """Makes a string prompt by joining all observations, one per line."""

    recent_memories_list = memory.retrieve_recent(5)
    recent_memories_set = set(recent_memories_list)
    recent_memories = "\n".join(recent_memories_set)

    relevant_memories_list = []
    for recent_memory in recent_memories_list:
        relevant =  memory.retrieve_associative(recent_memory, 3, add_time = False)
        for mem in relevant:
            if mem not in relevant_memories_list:
                relevant_memories_list.append(mem)

    relevant_memories = "\n".join(relevant_memories_list)
    return (
        f"\nYour recent memories are:\n{recent_memories}\n\n"
        f"Relevant memories from your past:\n{relevant_memories}\n\n"
    )

class SimpleLLMAgentWithAssociativeMemory(entity.Entity):
    def __init__(self, model: language_model.LanguageModel, embedder):
        self._model = model
        # The associative memory of the agent. It uses a sentence embedder to
        # retrieve on semantically relevant memories.
        self._memory = associative_memory.AssociativeMemory(embedder)

    @property
    def name(self) -> str:
        return 'Alice'

    def act(self, action_spec = entity.ActionSpec) -> str:
        prompt = make_prompt_associative_memory(self._memory)
        print(f"*****\nDEBUG: {prompt}\n*****")
        return self._model.sample_text(
            "You are a person.\n"
            f"Your name is {self.name}.\n"
            f"{prompt}\n"
            "What should you do next?"
        )
    
    def observe(self, observation: str) -> None:
        # Push a new observation into the memory, if there are too many, the oldest
        # one will be automatically dropped.
        self._memory.add(observation)


def get_agent(model, embedder):
    agent = SimpleLLMAgentWithAssociativeMemory(model, embedder)
    return agent