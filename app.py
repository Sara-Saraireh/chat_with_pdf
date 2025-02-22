import os
import streamlit as st
from langchain.llms import HuggingFaceHub

# Load Hugging Face API token securely
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Handle missing API key
if hf_token is None:
    st.error("❌ Hugging Face API token is missing! Set it in Streamlit Secrets or GitHub Secrets.")
    st.stop()

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Free LLM
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256},
    huggingfacehub_api_token=hf_token
)

# Streamlit UI
st.set_page_config(page_title="Free LLM Chatbot", page_icon="🤖")
st.title("🤖 Free AI Chatbot with Hugging Face")

st.markdown("### Ask me anything! (Powered by Mistral-7B)")
user_input = st.text_input("Your Question:", placeholder="Type here...")

if user_input:
    with st.spinner("Thinking... 💭"):
        response = llm(user_input)
    st.success("✅ Answer:")
    st.write(response)

# Footer
st.markdown("---")
st.markdown("💡 *Powered by Hugging Face & LangChain | Made with ❤️ in Streamlit*")
