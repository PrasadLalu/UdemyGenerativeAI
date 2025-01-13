import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load env data
load_dotenv()

# Access GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Access HuggingFace API Key
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Create model
model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
        Answer the question based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
            {context}
        </context>
        
        Question: {input}
    """
)

def create_vector_embeddings():
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("research_papers")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    

# App title
st.title("RAG Document Q&A with GROQ and Llama3")
user_prompt = st.text_input("Enter your query from the research paper.")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector database is ready.")
    
if user_prompt:
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    start_time = time.process_time()
    response = retriever_chain.invoke({ "input": user_prompt })
    print(f"Response Time: {time.process_time()-start_time}")
    
    st.write(response["answer"])
    
    # With a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)    
