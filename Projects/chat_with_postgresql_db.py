import os
import streamlit as st
from sqlalchemy import create_engine

from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Set page configuration first
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜", layout="centered")

# App setup
st.title("🦜 LangChain: Chat with SQL DB")

# Postgres credential
postgres_host="localhost"
postgres_user="postgres"
postgres_password="postgres"
postgres_db="collectiondb"
    
# Access Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
    
# LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(pg_host=None, pg_user=None, pg_password=None, pg_db=None):
    if not (pg_host and pg_user and pg_password and pg_db):
        st.error("Please provide all PostgreSQL connection details.")
        st.stop()
    return SQLDatabase(create_engine(f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}/{pg_db}"))
    

db = configure_db(postgres_host, postgres_user, postgres_password, postgres_db)

# Create database toolkit
toolkit = SQLDatabaseToolkit(llm=llm, db=db)

# Create sql agent
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.button("Clear message history"):
    st.session_state["messages"] = [{ "role": "assistant", "content": "How can I help you?" }]

for massage in st.session_state.messages:
    st.chat_message(massage["role"]).write(massage["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({ "role": "user", "content": user_query })
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = sql_agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({ "role": "assistant", "content": response })
        st.write(response)
