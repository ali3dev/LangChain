# helper.py
import os
import json
import re
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import AIMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERP_API_KEY = os.getenv("SERPER_API_KEY")

# Initialize the Gemini LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
embeddings = OpenAIEmbeddings()

def search_serp(query):
    """Search for news articles using the Serper API."""
    search = GoogleSerperAPIWrapper(k=5, type="search")
    response_json = search.results(query)
    return response_json

def pick_best_articles_urls(response_json, query):
    """Use the LLM to pick the best 3 article URLs from search results."""
    response_str = json.dumps(response_json)
    template = """
    You are an expert researcher. Here is a list of search results:
    {response_str}
    Select the best 3 articles and return ONLY a list of URLs in JSON format like: ["url1", "url2", "url3"]
    """
    prompt_template = PromptTemplate(input_variables=["response_str"], template=template)
    article_chooser_chain = prompt_template | llm
    urls_response = article_chooser_chain.invoke({"response_str": response_str})
    if isinstance(urls_response, AIMessage):
        urls_response = urls_response.content
    urls_response = re.sub(r"```json\s*|\s*```", "", urls_response)
    try:
        urls_list = json.loads(urls_response)
        return urls_list
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON response: {urls_response}")

def extract_content_from_urls(urls):
    """Extract content from the selected URLs and build a FAISS vector store using from_texts with debug prints."""
    print("Starting URL data extraction...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    print("Data loaded from URLs.")
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    docs = [doc for doc in docs if doc.page_content.strip()]
    print(f"Extracted {len(docs)} non-empty document chunks.")
    
    if not docs:
        raise ValueError("No valid document content extracted from URLs.")
    
    texts = [doc.page_content for doc in docs]
    limited_texts = texts[:5]
    print(f"Using {len(limited_texts)} chunks for FAISS index (limited for debugging).")
    
    # Dummy FAISS test
    try:
        dummy_db = FAISS.from_texts(["Test chunk"], embeddings)
        print("Dummy FAISS index built successfully.")
    except Exception as e:
        print("Error building dummy FAISS index:", e)
    
    try:
        db = FAISS.from_texts(limited_texts, embeddings)
        print("FAISS index built successfully with limited texts.")
    except Exception as e:
        print("Error building FAISS index:", e)
        raise e
    
    return db

def summarizer(db, query, k=4):
    """Summarize the content using the LLM based on similarity search."""
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    template = """
    {docs}
    Summarize the above content into an engaging newsletter on {query}.
    """
    prompt_template = PromptTemplate(input_variables=["docs", "query"], template=template)
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = summarizer_chain.run(docs=docs_page_content, query=query)
    return summary.replace("\n", "")

def generate_newsletter(summaries, query):
    """Generate the final newsletter using the LLM."""
    template = """
    {summaries}
    Write a professional and engaging newsletter on {query} in an informal style.
    """
    prompt_template = PromptTemplate(input_variables=["summaries", "query"], template=template)
    newsletter_chain = LLMChain(llm=llm, prompt=prompt_template)
    newsletter_text = newsletter_chain.predict(summaries=summaries, query=query)
    return newsletter_text
