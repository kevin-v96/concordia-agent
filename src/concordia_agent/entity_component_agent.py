from concordia.typing import entity_component, entity
from concordia.agents import entity_agent
from concordia.memory_bank import legacy_associative_memory
from concordia.associative_memory import associative_memory
from concordia.language_model import language_model
from concordia.components.agent import memory_component, action_spec_ignored

class Observe(entity_component.ContextComponent):
    def pre_observe(self, observation: str) -> str:
        self.get_entity().get_component('memory').add(observation, {})

class RecentMemories(action_spec_ignored.ActionSpecIgnored):
    def __init__(self):
        super().__init__('Recent Memories')

    def _make_pre_act_value(self) -> str:
        recent_memories_list = self.get_entity().get_component('memory').retrieve(
            query='',  # Don't need a query to retrieve recent memories.
            limit=5,
            scoring_fn=legacy_associative_memory.RetrieveRecent(),
        )
        recent_memories = " ".join(memory.text for memory in recent_memories_list)
        print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
        return recent_memories

def _recent_memories_str_to_list(recent_memories: str) -> list[str]:
    # Split sentences, strip whitespace and add final period
    return [memory.strip() + '.' for memory in recent_memories.split('.')]

class RelevantMemories(action_spec_ignored.ActionSpecIgnored):
    def __init__(self):
        super().__init__('Relevant Memories')

    def _make_pre_act_value(self) -> str:
        recent_memories = self.get_entity().get_component('recent_memories').get_pre_act_value()
        # Each sentence will be used for retrieving new relevant memories.
        recent_memories_list = _recent_memories_str_to_list(recent_memories)
        recent_memories_set = set(recent_memories_list)

        memory = self.get_entity().get_component('memory')
        relevant_memories_list = []
        for recent_memory in recent_memories_list:
            # Retrieve 3 memories that are relevant to the recent memory.
            relevant = memory.retrieve(
                query = recent_memory,
                limit = 3,
                scoring_fn = legacy_associative_memory.RetrieveAssociative(add_time=False)
            )
            for mem in relevant:
                # Make sure that we only add memories that are _not_ already in the recent
                # ones.
                if mem.text not in recent_memories_set:
                    relevant_memories_list.append(mem.text)
                    recent_memories_set.add(mem.text)

        relevant_memories = "\n".join(relevant_memories_list)
        print(f"*****\nDEBUG: Relevant memories:\n{relevant_memories}\n*****")
        return relevant_memories

    
class SimpleActing(entity_component.ActingComponent):
    def __init__(self, model: language_model.LanguageModel):
        self._model = model

    def get_action_attempt(
            self, contexts, action_spec = entity.DEFAULT_ACTION_SPEC
    ) -> str:
        # Put context from all components into a string, one component per line.
        context_for_action = '\n'.join(
            f"{name}: {context}" for name, context in contexts.items()
        )
        print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n******")
        # Ask the LLM to suggest an action attempt.
        call_to_action = action_spec.call_to_action.format(
            name = self.get_entity().name, timedelta = "2 minutes"
        )

        return self._model.sample_text(
            f"{context_for_action}\n\n{call_to_action}\n"
        )
def get_agent(model, embedder):
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(
        associative_memory.AssociativeMemory(embedder)
    )

    # Let's create an agent with the above components.
    agent = entity_agent.EntityAgent(
        "Alice",
        act_component=SimpleActing(model),
        context_components={
            "observation": Observe(),
            'relevant_memories': RelevantMemories(),
            "recent_memories": RecentMemories(),
            "memory": memory_component.MemoryComponent(raw_memory),
        }
    )

    return agent