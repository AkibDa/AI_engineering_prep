import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

load_dotenv()

st.title("Restaurent Name Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ["Italian", "Chinese", "Mexican", "Indian", "French"])

def generate_restaurant_name_and_items(cuisine):
  prompt_template_name = PromptTemplate(
      input_variables=["cuisine"],
      template="Generate a creative and catchy name for a {cuisine} restaurant."
  )
  name_chain = LLMChain(
      llm=init_chat_model("google_genai:gemini-2.5-flash-lite"),
      prompt=prompt_template_name,
      output_key="Name"
  )
  prompt_template_menu = PromptTemplate(
      input_variables=["Name"],
      template="Generate a list of 5 popular menu items for a restaurant named {Name}."
  )
  menu_chain = LLMChain(
      llm=init_chat_model("google_genai:gemini-2.5-flash-lite"),
      prompt=prompt_template_menu,
      output_key="Menu_Items"
  )
  sequential_chain = SequentialChain(
      chains=[name_chain, menu_chain],
      input_variables=["cuisine"],
      output_variables=["Name", "Menu_Items"]
  )
  resp = sequential_chain({"cuisine": cuisine})
  return resp

if cuisine:
  result = generate_restaurant_name_and_items(cuisine)
  st.header(result["Name"].strip())
  menu_items = result["Menu_Items"].strip().split(",")
  st.subheader("Menu Items:")
  for item in menu_items:
      st.write(f"- {item.strip()}")
