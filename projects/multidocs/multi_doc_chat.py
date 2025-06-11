# multi_doc_chat.py
import os 
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import
from load_docs import load_docs
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message  # pip install streamlit_chat

# Load environment variables
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=GOOGLE_API_KEY)

# Load documents from the 'docs' folder
documents = load_docs()
if not documents:
    st.error("No documents loaded. Please add documents to the 'docs' folder.")
    st.stop()
chat_history = []

# Split documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
if not docs:
    st.error("Document splitting resulted in no chunks. Check your documents.")
    st.stop()

# Initialize embeddings with API key & model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # ✅ Correct model name
    google_api_key=GOOGLE_API_KEY   # ✅ API key
)

# Create our vector database using FAISS (no persistence by default)
vectordb = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

# Set up Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# ==== Streamlit front-end ====
st.title("Docs QA Bot using Langchain (FAISS)")
st.header("Ask anything about your documents... 🤖")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_query():
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text

# Retrieve the user input and display chat messages
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        # User message first
        message(st.session_state['past'][i], is_user=True, key=f"user_{i}")
        # Bot answer second
        message(st.session_state['generated'][i], key=f"bot_{i}")