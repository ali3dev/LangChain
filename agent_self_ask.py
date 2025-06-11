import os 
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools

from langchain.agents import Tool, initialize_agent, load_tools
from langchain import SerpAPIWrapper

# Load API Key
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY") 

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.0)

search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)

# tools
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="google search"
    )
]

# initialize our agent
self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent='self-ask-with-search',
    verbose=True
)

query = "who has travelled the most: Justin Timberlake, Alicia Keys, or Jason Mraz?"
result = self_ask_with_search(query)