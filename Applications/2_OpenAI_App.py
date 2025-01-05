import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env data
load_dotenv(find_dotenv())

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response the question asked."),
        ("user", "{question}")
    ]
)

# Streamlit
st.title("Langchain Demo with GPT Model")
input_text = st.text_input("What's in your mind?")

# Initialize model
llm = ChatOpenAI(model="gpt-4o")

# Create output parse
outout_parser = StrOutputParser()

# Create chain
chain = prompt | llm | outout_parser

if input_text:
    st.write(chain.invoke({ input_text }))
