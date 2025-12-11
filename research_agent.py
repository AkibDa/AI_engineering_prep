import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent  # <--- The new unified import
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

_ = load_dotenv()

# 1. Setup Model
llm=init_chat_model("google_genai:gemini-2.5-flash-lite")


# 2. Define Output Schema
class ResearchResponse(BaseModel):
  topic: str = Field(description="The topic of research")
  summary: str = Field(description="A detailed summary of the findings")
  sources: list[str] = Field(description="List of URLs or sources used")


# 3. Define Tools
search = DuckDuckGoSearchRun()


@tool
def fetch_web_content(url: str) -> str:
  """Fetch and summarize content from a webpage."""
  return f"Simulated content for {url} (Real implementation would use WebBaseLoader)"


tools = [search, fetch_web_content]

# 4. Create the Agent (New v1.0+ Syntax)
# Note: create_agent now handles the "loop" internally (replacing AgentExecutor)
agent = create_agent(
  model=llm,
  tools=tools,
  system_prompt="You are a researcher. Use tools to gather info, then return structured JSON.",
  response_format=ResearchResponse  # Built-in structured output support
)

# 5. Run it
print("--- Starting Research ---")
try:
  # The new agent returns the structured object directly in the 'structured_response' key
  # or directly depending on configuration
  result = agent.invoke({
    "messages": [{"role": "user", "content": "Research the impact of AI on coding jobs"}]
  })

  # Extract structured response
  structured_response: ResearchResponse = result['structured_response']
  print("Research Topic:", structured_response.topic)
  print("Summary:", structured_response.summary)
  print("Sources:", structured_response.sources)

except Exception as e:
  print(f"Error: {e}")