import os
import streamlit as st
from langchain_groq import ChatGroq

from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMMathChain, LLMChain
from langchain.callbacks import StreamlitCallbackHandler

# Setup the streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Using Google Gemma2")

# Access Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Create model
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

# Initial wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various information on the topic mentioned."
)

# Initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to bed provided."
)

prompt = """
    Your a agent tasked for solving users mathematical question. Logically arrive at the solution and provide 
    a detailed explanation and display it point wise for the question below
    Question:{question}
    Answer:
"""

# Prompt
prompt_template = PromptTemplate(
    template=prompt,
    input_variables=["question"],
)

# Combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Create reasoning tool
reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agents
assistant_agent=initialize_agent(
    llm=llm,
    verbose=False,
    handle_parsing_errors=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[wikipedia_tool, math_tool, reasoning_tool]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a Math chatbot who can answer all your maths questions"
        }
    ]

for massage in st.session_state.messages:
    st.chat_message(massage["role"]).write(massage['content'])
    
# Let's start the interaction
question = st.text_area("Enter your question:","st_callback")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_callback=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_callback])
            st.session_state.messages.append({'role':"assistant", "content": response})
            st.write("### Response:")
            st.success(response)
    else:
        st.warning("Please enter the question")
