from crewai import Agent
from tools import youtube_tool
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

# Correctly set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"]="gpt-4-0125-preview"

llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model=os.environ["OPENAI_MODEL_NAME"])

# Create a blog content researcher
blog_researcher = Agent(
    role="Blog researcher from Youtube videos",
    goal="Get relevant video content for the {topic} from the youtube channel",
    verbose=True,
    memory=True,
    backstory=(
        "Export in understanding videos in AI Data science, Machine learning, generative AI and provide suggestion"
    ),
    tools=[youtube_tool],
    allow_delegation=False,
    llm=llm    
)

# Create a blog writer agent
blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic} from Youtube channel",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new",
        "discoveries to light in an accessible manner."   
    ),
    tools=[youtube_tool],
    allow_delegation=False
)