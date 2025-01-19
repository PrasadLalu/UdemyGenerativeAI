import sqlite3
import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine

from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

LOCAL_DB="LOCAL_DB"
POSTGRESQL="POSTGRESQL"

# Set page configuration first
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜", layout="centered")

# App setup
st.sidebar.title("Setting")
st.title("🦜 LangChain: Chat with SQL DB")

radio_options = ["Use SQLite 3 Database - student.db", "Connect to PostgreSQL Database"]
selected_option = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_options)

if radio_options.index(selected_option) == 1:
    db_uri=POSTGRESQL
    postgres_host=st.sidebar.text_input("PostgreSQL Host")
    postgres_user=st.sidebar.text_input("PostgreSQL User")
    postgres_password=st.sidebar.text_input("PostgreSQL password", type="password")
    postgres_db=st.sidebar.text_input("PostgreSQL database")
else:
    db_uri=LOCAL_DB
    
groq_api_key = st.sidebar.text_input(label="Enter Groq API Key", type="password")

if not db_uri:
    st.info("Please enter the database information and URI")

if not groq_api_key:
    st.info("Please provide Groq API Key")
    
# LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, pg_host=None, pg_user=None, pg_password=None, pg_db=None):
    if db_uri == "LOCAL_DB":
        db_file_path = (Path(__file__).parent/"student.db").absolute()
        print(db_file_path)
        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == "POSTGRESQL":
        if not (pg_host and pg_user and pg_password and pg_db):
            st.error("Please provide all PostgreSQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}/{pg_db}"))
    else:
        st.error("Invalid database type specified.")
        st.stop()

if db_uri == "POSTGRESQL":
    db = configure_db(db_uri, postgres_host, postgres_user, postgres_password, postgres_db)
else:
    db = configure_db(db_uri)

# Create database toolkit
toolkit = SQLDatabaseToolkit(llm=llm, db=db)

# Create sql agent
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
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
