from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import AgentType, initialize_agent, load_tools
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory

load_dotenv()

llm=init_chat_model("google_genai:gemini-2.5-flash-lite")

memory=ConversationBufferWindowMemory(k=1)
conversation_chain=ConversationChain(llm=llm, memory=memory)

tools = load_tools(["wikipedia","llm-math"], llm=llm, memory=memory)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
response = agent.run("What is the birth date of Albert Einstein? Also, what is 15% of that year?")
print(response)