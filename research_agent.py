from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model


load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")