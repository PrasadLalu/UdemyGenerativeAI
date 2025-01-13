import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env data
load_dotenv()

# Access GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response the query based on your best ability."),
        ("user", "Question: {question}")
    ]
)

# Create output parser
output_parser = StrOutputParser()

# App title
st.title("Q&A Chatbot with GRAQ API")

# Sidebar setting
st.sidebar.title("Setting")

# GROQ models
selected_model = st.sidebar.selectbox("Select a model", ["llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it", "whisper-large-v3", "whisper-large-v3-turbo"])

# Temperatures
temperatures = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Max tokens
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# User input
user_input = st.text_input("Go ahead and asked your query: ")

def generate_response(question, model_name):
    model = ChatGroq(model=model_name, groq_api_key=groq_api_key)
    
    chain = prompt | model | output_parser
    response = chain.invoke({ "question": question })
    return response
    

if user_input:
    answer = generate_response(user_input, selected_model)
    st.write(answer)
else:
    st.write("Please enter your query.")
