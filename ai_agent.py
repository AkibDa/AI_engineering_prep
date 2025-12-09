from dataclasses import dataclass
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
  user_id: str

@dataclass
class ResponseFormat:
  summary: str
  temperature_celsius: float
  temperature_fahrenheit: float
  humidity: float
  wind_speed: float

@tool('get_weather', description='Return weather information for a given city.', return_direct=False)
def get_weather(city: str) -> str:
    """Fetches weather information for a specified city using a public API."""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Look up a user's city based on the context ", return_direct=False)
def locate_user(runtime: ToolRuntime[Context]):
  match runtime.context.user_id:
    case 'ABC123':
      return 'Vienna'
    case 'XYZ456':
      return 'London'
    case 'HJK111':
      return 'Paris'
    case _:
      return 'Unknown'

checkpointer = InMemorySaver()

agent = create_agent(
  model=init_chat_model("google_genai:gemini-2.5-flash-lite"),
  tools=[get_weather, locate_user],
  system_prompt="You are a helpful assistant that provides weather information, who always cracks a joke related to weather.",
  response_format=ResponseFormat,
  checkpointer=checkpointer,
  context_schema=Context,
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
  'messages': [
    {'role': 'user', 'content': 'What is the weather like in New York City today?'}
  ]},
  config= config,
  context= Context(user_id='ABC123')
)

print(response['structured_response'].summary)