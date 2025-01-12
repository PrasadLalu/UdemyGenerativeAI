import os
from dotenv import load_dotenv, find_dotenv

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env data
load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please give proper answer to user query."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, llm, api_key, temperatures, max_tokens):
    openai.api_key = api_key
    model = ChatOpenAI(model=llm)
    
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    answer = chain.invoke({ "question": question })
    return answer
    

# Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

# Sidebar setting
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# OpenAI models
model = st.sidebar.selectbox("Select OpenAI model: ", ["gpt-4", "gpt-4o", "gpt-4-turbo"])

# Adjust response parameters
temperatures = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# User input
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, model, api_key, temperatures, max_tokens)
    st.write(response)
