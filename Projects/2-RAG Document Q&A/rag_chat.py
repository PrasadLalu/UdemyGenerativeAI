import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.memory import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load env data
load_dotenv()

# Access GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Access HuggingFace API Key
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Create embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# App title
st.title("Conversational RAG With PDF Upload AND Chat History")
st.write("Upload PDF and Chat With Their Content")

# Groq API Key
api_key = st.text_input("Enter Groq API Key: ", type="password")

if api_key:
    # Create model
    llm = ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
    
    # Chat interface
    session_id = st.text_input("Session ID", value="default_session")
    
    if "store" not in st.session_state:
        st.session_state.store = {}
        
    # Upload PDF file
    uploaded_files=st.file_uploader("Upload A PDf file", type="pdf", accept_multiple_files = True)
    
    # Process PDF file
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
            st.success(f"File '{file_name}' uploaded and processed.")
        
        # Split and create document embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,  persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()  
        
        contextualize_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        ) 
        
        history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualize_prompt)

        # Question-Answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )
        
        user_query = st.text_input("Enter you query: ")
        if user_query:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                { "input": user_query },
                config={"configurable": {"session_id": session_id}}
            )
            
            st.write(st.session_state.store)
            st.write("Assistant: ", response["answer"])
            st.write("Chat History: ", session_history.messages)
else:
    st.write("Please enter Groq API Key.")
