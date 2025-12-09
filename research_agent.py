from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Create the system prompt as a string with format instructions
system_prompt = f"""
You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text
{parser.get_format_instructions()}
"""

agent = create_agent(model=llm, system_prompt=system_prompt)

# Invoke the agent with proper message format
raw_response = agent.invoke({
    "messages": [HumanMessage(content="Provide a detailed summary on the impact of climate change on marine biodiversity.")]
})

# Parse the output using your parser
if isinstance(raw_response, dict) and "messages" in raw_response:
    # Extract the last message content from the agent response
    response_text = raw_response["messages"][-1].content
    parsed_response = parser.parse(response_text)
else:
    # Fallback for different response formats
    parsed_response = parser.parse(str(raw_response))

print(parsed_response.summary)
