import sentence_transformers
import os
import dotenv

from concordia_agent.language_model import gpt_model

from concordia_agent.simple_agent import get_agent

dotenv.load_dotenv()

_embedder_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
embedder = lambda x: _embedder_model.encode(x, show_progress_bar=False)


if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

model = gpt_model.GptLanguageModel(api_key = os.environ.get("OPENAI_API_KEY"), model_name = os.environ.get("OPENAI_MODEL"))

agent = get_agent(model)
agent.observe("You are in a room.")
agent.observe("The room has only a table in it.")
agent.observe("On the table there is a single apple.")
agent.observe("The apple is shinny red and looks absolutely irresistible!")
print(agent.act())