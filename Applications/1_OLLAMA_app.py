import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env data
load_dotenv(find_dotenv())

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistent. Please response the question asked"),
        ("user", "Question: {question}")
    ]
)

st.title("Langchain Demo with Gemma")
input_text = st.text_input("What question you have in you mind?")

# Initialize LLM model
llm = OllamaLLM(model='gemma2:2b')

# Create Output parser
output_parser = StrOutputParser()

# Create chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({ "question: {input_text}"}))

