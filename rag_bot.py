from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

texts = [
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is a historic fortification.",
    "The Statue of Liberty is in New York City.",
    "Machu Picchu is an ancient Incan city in Peru.",
    "The Colosseum is a large amphitheater in Rome."
]

vectorstore = FAISS.from_texts(texts, embeddings)

query = "Where is the Eiffel Tower located?"
docs = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(docs):
    print(f"Document {i+1}: {doc.page_content}")

query_2 = "What is Machu Picchu?"
docs_2 = vectorstore.similarity_search(query_2, k=2)

for i, doc in enumerate(docs_2):
    print(f"Document {i+1}: {doc.page_content}")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Create a custom retriever tool using @tool decorator
@tool
def landmark_retriever(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "No documents found."

retriever_tool = landmark_retriever

# Create the model
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

print(f"Tool created: {landmark_retriever}")
print(f"Tool name: {landmark_retriever.name}")
print(f"Tool description: {landmark_retriever.description}")

# Create the agent with system prompt
agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant that provides information about famous landmarks using the landmark_retriever tool. First call the landmark_retriever tool to get relevant information, then answer the user's question based on that information."
)

print("Agent created successfully!")

# Invoke the agent with proper message format
print("Invoking agent...")
result = agent.invoke({
    "messages": [HumanMessage(content="Tell me about the Eiffel Tower.")]
})

print("Agent Result:")
if isinstance(result, dict) and "messages" in result:
    print(result["messages"][-1].content)
else:
    print(result)
