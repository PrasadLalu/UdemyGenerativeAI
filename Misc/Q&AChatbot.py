import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load end data
load_dotenv()

# Access OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create model
model = ChatOpenAI(model="gpt-4o")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant. Please give response to asked query."),
        ("user", "Question: {question}")
    ]
)

# Create Output Parser
output_parser = StrOutputParser()

# Create chain
chain = prompt | model | output_parser

# Invoke chain


st.title("Q&A Chatbot with OpenAI")
input_text = st.text_input("Asked you question: ")

if input_text:
    answer = chain.invoke({ "question": "What is generative ai?"})
    st.write(answer)
    

