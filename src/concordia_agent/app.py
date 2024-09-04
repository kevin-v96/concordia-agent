import sentence_transformers
import os
import dotenv
import warnings

warnings.filterwarnings("ignore")

from concordia.language_model import gpt_model

from concordia_agent.simple_agent_with_associative_memory import get_agent

dotenv.load_dotenv()

_embedder_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)


if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

model = gpt_model.GptLanguageModel(api_key = os.environ.get("OPENAI_API_KEY"), model_name = os.environ.get("OPENAI_MODEL"))
agent = get_agent(model, embedder)

agent.observe("You absolutely hate apples and would never willingly eat them.")
agent.observe("You don't particularly like bananas.")
# Only the next 5 observations will be retrieved as "recent memories"
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there are two fruits: an apple and a banana.")
agent.observe("The apple is shiny red and looks absolutely irresistible!")
agent.observe("The banana is slightly past its prime.")
print(agent.act())