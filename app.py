# Import required libraries
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.serpapi import SerpApiTools
from textwrap import dedent
import pyperclip
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

# Set up the Streamlit app
st.title("AI Journalist Agent 🗞️")
st.caption("Generate high-quality articles with AI Journalist")

# Initialize the AI journalist agent
journalist = Agent(
    name="AI Journalist",
    role="A world-class journalist who writes compelling news articles",
    model=Groq(id="mixtral-8x7b-32768", api_key=groq_api_key),
    description=dedent(
        """\
        You are a senior journalist for a major news outlet. Given a topic,
        your goal is to write a high-quality, well-researched article by:
        1. Searching for relevant and recent information using Serper
        2. Analyzing the information and identifying key points
        3. Writing a comprehensive, engaging article with proper attribution
        """
    ),
    instructions=[
        "Search for recent and relevant information about the topic using Serper",
        "Write a well-structured article with a compelling headline",
        "Include relevant quotes and statistics when available",
        "Maintain journalistic integrity and fact-check information",
        "Provide balanced coverage of different viewpoints when applicable",
        "Use clear, engaging language suitable for a general audience",
        "Ensure proper attribution for sources and quotes",
    ],
    tools=[SerpApiTools(api_key=serper_api_key)],
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)

# Initialize LinkedIn post creator agent
linkedin_creator = Agent(
    name="LinkedIn Creator",
    role="A professional LinkedIn content creator",
    model=Groq(id="mixtral-8x7b-32768", api_key=groq_api_key),
    description="You are a professional LinkedIn content creator who creates engaging posts from articles.",
    instructions=[
        "Create a professional LinkedIn post from the given article",
        "Keep the post between 1000-1300 characters",
        "Include 3-5 relevant hashtags",
        "Add line breaks for readability",
        "End with an engaging question or call to action",
        "Maintain a professional tone throughout",
    ],
    markdown=True,
)

# Input field for the article topic
topic = st.text_input("What topic would you like an article about?")

if topic:
    with st.spinner("Researching and writing your article..."):
        try:
            # Generate the article without streaming
            response = journalist.run(
                f"Write a comprehensive news article about: {topic}",
                stream=False
            )
            
            # Display the article
            st.markdown(response.content)
            
            # Store the generated article
            st.session_state['generated_article'] = response.content
            
            # Add LinkedIn post generation button
            if st.button("Generate LinkedIn Post"):
                with st.spinner("Creating LinkedIn post..."):
                    # Generate LinkedIn post using the LinkedIn creator agent
                    linkedin_response = linkedin_creator.run(
                        f"""Create a professional LinkedIn post based on this article:
                        
                        {response.content}
                        
                        Remember to include hashtags, maintain readability with line breaks, 
                        and end with an engaging question.""",
                        stream=False
                    )
                    
                    # Display the LinkedIn post in a special format
                    st.subheader("📱 LinkedIn Post")
                    st.info(linkedin_response.content)
                    
                    # Add copy button
                    if st.button("Copy to Clipboard"):
                        try:
                            pyperclip.copy(linkedin_response.content)
                            st.success("Post copied to clipboard! You can now paste it on LinkedIn.")
                        except Exception as e:
                            st.error(f"Could not copy to clipboard: {str(e)}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again with a different topic.")