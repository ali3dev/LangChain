import os
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API Key from .env
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google Gemini AI Model
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                              temperature=0.7,
                              google_api_key=GOOGLE_API_KEY)

# Define Prompt Template
template = """ 
As a children's book writer, please come up with a simple and short (90 words)
lullaby based on the location {location} and the main character {name}.

STORY:
"""

prompt = PromptTemplate(input_variables=["location", "name"], template=template)

# Create LLMChain
chain_story = LLMChain(llm=chat, prompt=prompt, verbose=True)

# Generate Story
story = chain_story({"location": "Zanzibar", "name": "Maya"})

# Print the generated story
print(story['text'])