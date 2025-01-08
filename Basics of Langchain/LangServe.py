# Load end data
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Access Groq api key
groq_api_key = os.getenv("GROQ_API_KEY")

# Create model
model =  ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Create chat template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user","{text}")
    ]
)

# Create output parser
output_parser = StrOutputParser()

# Create chain
chain = prompt_template | model | output_parser

# Defined app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces"
)

# Add routes
add_routes(
    app,
    chain,
    path="/chain"    
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
