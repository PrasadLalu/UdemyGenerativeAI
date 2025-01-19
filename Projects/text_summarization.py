import os
import validators
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from urllib.parse import urlparse, parse_qs
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader

# Create streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

groq_api_key = os.getenv("GROQ_API_KEY")

# Gemma model
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

# Prompt
prompt_template = """
    Provide the summary of the following content in 300 wards:
    Contents: {text}
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

generic_url = st.text_input("URL", label_visibility="collapsed")

def extract_video_id(video_url):
    """
    Extract the video ID from the YouTube URL.
    """
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    if not video_id:
        raise ValueError("Invalid YouTube URL: Unable to extract video ID.")
    return video_id
   
if st.button("Summarize the content of Youtube or Website"):
        if not groq_api_key.strip() or not generic_url.strip():
            st.error("Please provide information to get started")
        elif not validators.url (generic_url):
            st.error("Please enter a valid url. It can may be a YT url or Website url")
        else:
            try:
                with st.spinner("Waiting..."):
                    if "youtube.com" in generic_url:
                        # Extract video ID
                        video_id = extract_video_id(generic_url)

                        # Initialize the YouTube loader
                        loader = YoutubeLoader(video_id, add_video_info=False)

                        # Load documents (transcript and metadata)
                        docs = loader.load()

                        # Load the summarization chain
                        chain = load_summarize_chain(llm, chain_type="stuff")

                        # Generate summary
                        summary = chain.invoke(docs)
                        st.success(summary)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url], 
                            ssl_verify=False, 
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                        
                        # Load documents (transcript and metadata)
                        docs = loader.load()
                        
                        # Summarize the text
                        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.invoke(docs) 
                        st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception {e}")

