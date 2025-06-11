import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Selection
API_PROVIDER = os.getenv("API_PROVIDER", "GOOGLE")  # "GOOGLE" or "OPENAI"

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
