import os 
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm_model = "gemini-1.5-flash"
chat  = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)

## 1. CharacterTextSplitter
with open("./data/i-have-a-dream.txt", encoding="utf-8") as paper:
    speech = paper.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len,
    add_start_index=True
)

docs = text_splitter.create_documents([speech])
print(docs[0])


#====== Test ======


# s1 = "abcdefghijklmnopqrstuvwxyz"
# s = "Python can be easy to pick up whether you're a professional or a beginner."

# text = text_splitter.split_text(s)
# print(text)