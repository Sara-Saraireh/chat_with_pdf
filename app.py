import streamlit as st
from langchain.llms import HuggingFaceHub
import os

# Load API key securely
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Initialize the LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

st.title("Free LLM Chatbot")
user_input = st.text_input("Ask something:")
if user_input:
    response = llm(user_input)
    st.write(response)
