from crewai import Task
from tools import youtube_tool
from agents import blog_researcher, blog_writer

# Create research task
research_task = Task(
    description=(
        "Identify the video {topic}",
        "Get details information about the video from the channel."
    ),
    expected_output="A comprehensive 3 paragraphs long report based on the {topic} of video content.",
    tools=[youtube_tool],
    agent=blog_researcher
)

# Create write task
write_task = Task(
    description=(
        "Get the information from youtube channel on the topic {topic}."
    ),
    expected_output="Summarize the information from the youtube channel video topic on the {topic} and create the content for the blog.",
    tools=[youtube_tool],
    agent=blog_writer,
    async_execution=False,
    output_file="new-blog-post.md"
)