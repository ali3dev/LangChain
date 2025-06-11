import os 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Step 1: Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Step 2: Load same embeddings used during creation
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Step 3: Load vector database
db = Chroma(persist_directory=persistent_directory, 
            embedding_function=embeddings)

# Step 4: Ask a question
query = "What is the Odyssey about?"

# Step 5: Retrieve top matching results (with threshold filtering)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.9},
)

relevant_docs = retriever.invoke(query)

# Step 6: Show Results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
