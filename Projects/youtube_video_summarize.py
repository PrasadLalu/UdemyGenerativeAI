
import os
import streamlit as st
from langchain_groq import ChatGroq
from urllib.parse import urlparse, parse_qs
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader


# Create streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Prompt
prompt_template = """
    Provide the summary of the following content in 300 wards:
    Contents: {text}
"""
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

def extract_video_id(video_url):
    """
    Extract the video ID from the YouTube URL.
    """
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    if not video_id:
        raise ValueError("Invalid YouTube URL: Unable to extract video ID.")
    return video_id


def summarize_youtube_video(video_url):
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)

        # Initialize the YouTube loader
        loader = YoutubeLoader(video_id, add_video_info=False)

        # Load documents (transcript and metadata)
        docs = loader.load()

        # Load the summarization chain
        chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)

        # Generate summary
        summary = chain.run(docs)
        return summary

    except Exception as e:
        return f"An error occurred: {e}"

input_url = st.text_input("Enter Youtube Video URL Here: ")

if st.button("Summarize the content of Youtube or Website"):
    try:
        with st.spinner("Processing..."):
            if "youtube.com" in input_url:
                # Summarize the video
                summary = summarize_youtube_video(input_url)
                st.success(summary)
            else:
                loader = UnstructuredURLLoader(
                            urls=[input_url], 
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
    