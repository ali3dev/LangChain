import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Updated imports for Gemini integration
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Fixed import
from langchain_community.document_loaders import PyPDFLoader  # If still using community for loaders
from langchain_community.vectorstores import Chromadb as Chroma  # If still using community for vectorstores
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load env variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)

# Initialize Gemini LLM (using Google AI SDK directly if needed)
llm = genai.GenerativeModel("gemini-1.5-flash")

# Initialize embeddings using the updated import
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)

# Load PDF document
loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Store embeddings in Chroma vector database
persist_directory = './data/db/chroma/'
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()

# Load vector store and make retriever
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Retrieve relevant documents
docs = retriever.get_relevant_documents("Tell me more about ReAct prompting")
print(docs[0].page_content)

# Create QA Chain with Gemini
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
)

# Helper function to format LLM response
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# Query the LLM
query = "tell me more about ReAct prompting"
llm_response = qa_chain(query)
process_llm_response(llm_response=llm_response)
