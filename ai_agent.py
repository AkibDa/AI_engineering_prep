import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Return weather information for a given city.', return_direct=False)
def get_weather(city: str) -> str:
    """Fetches weather information for a specified city using a public API."""
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
  model=init_chat_model("google_genai:gemini-2.5-flash-lite"),
  tools=[get_weather],
  system_prompt="You are a helpful assistant that provides weather information, who always cracks a joke related to weather.",
)

response = agent.invoke({
  'messages': [
    {'role': 'user', 'content': 'What is the weather like in New York City today?'}
  ]
})

print(response['messages'][-1].content)