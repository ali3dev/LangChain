# multi_doc_chat_fancy.py
# this is test ui file but there are some error occure later i will updated 

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message  # pip install streamlit_chat

from load_docs import load_docs  # Your existing load_docs.py

# -- Set up page config for a nicer look --
st.set_page_config(
    page_title="Docs QA Bot - Fancy Edition",
    page_icon=":sparkles:",
    layout="wide"
)

# -- Inject some custom CSS to style the chat bubbles & background --
custom_css = """
<style>
/* Overall page background color */
body {
    background-color: #F0F2F6;
    font-family: 'Helvetica Neue', sans-serif;
}

/* Main container to center the chat */
.main-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
}

/* Chat bubble styling */
.message-bubble {
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 12px;
    line-height: 1.4;
}

.user-message {
    background-color: #DCF8C6; /* Light green bubble */
    text-align: right;
}

.bot-message {
    background-color: #ECECEC; /* Light gray bubble */
    text-align: left;
}

/* Make the input box stand out a bit */
.css-1u8p2o7 {
    border-radius: 8px;
    border: 1px solid #CCC;
}

/* Hide Streamlit's default header/footer if desired */
.block-container {
    padding-top: 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -- Sidebar with instructions or brand info --
with st.sidebar:
    st.title("Fancy Docs QA Bot")
    st.write("Here you can tweak settings or add instructions.")
    st.write("---")
    st.write("**Instructions**")
    st.write("1. Add your docs to the `docs` folder.")
    st.write("2. Ask questions about the loaded documents.")
    st.write("3. Enjoy the fancy UI!")

# -- Load environment variables --
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -- Initialize the LLM --
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=GOOGLE_API_KEY)

# -- Load documents --
documents = load_docs()
if not documents:
    st.error("No documents loaded. Please add documents to your `docs` folder.")
    st.stop()

# -- Split the documents into chunks --
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
if not docs:
    st.error("Document splitting produced no chunks!")
    st.stop()

# -- Create embeddings & FAISS vector store --
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectordb = FAISS.from_documents(docs, embedding)

# -- Set up Conversational Retrieval Chain --
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

# -- Initialize session state for chat history --
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# -- Main page layout --
st.title("Docs QA Bot :sparkles:")
st.write("**Ask anything about your documents**")

def get_query():
    # Chat input is at the bottom of the page
    return st.chat_input("Type your question here...")

# -- Create a container to hold the chat conversation in the center --
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Display chat messages in custom bubbles
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            user_msg = st.session_state['past'][i]
            bot_msg = st.session_state['generated'][i]
            # User bubble
            st.markdown(
                f"<div class='message-bubble user-message'>{user_msg}</div>",
                unsafe_allow_html=True
            )
            # Bot bubble
            st.markdown(
                f"<div class='message-bubble bot-message'>{bot_msg}</div>",
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

# -- Capture user input & process it --
user_input = get_query()
if user_input:
    # Use "query" as the key and pass chat_history if you want context
    result = qa_chain.invoke({"query": user_input, "chat_history": st.session_state['chat_history']})
    answer = result.get("result", "")

    # Update session state
    st.session_state['chat_history'].append((user_input, answer))
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(answer)
