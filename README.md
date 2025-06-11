# LangChain Experiments

This repository contains hands-on experiments and examples using [LangChain](https://github.com/langchain-ai/langchain) for Retrieval-Augmented Generation (RAG) and Conversational Agents with memory and tool use.


---

## 🚀 Examples

### 1. Retrieval-Augmented Generation (RAG)

**File:** `RAG/1a_rag_basics.py`

- Loads a text file (`odyssey.txt`).
- Splits it into manageable chunks.
- Converts each chunk into embeddings using HuggingFace models.
- Stores embeddings in a Chroma vector database.
- Supports semantic search over the stored vectors.

**How to run:**
```bash
python RAG/1a_rag_basics.py
```

---

### 2. Conversational Agent with Tools & Memory

**File:** `agent_conversational.py`

- Loads your Google Gemini API key from `.env`.
- Initializes a Gemini LLM with LangChain.
- Adds conversational memory (remembers chat history).
- Loads tools (e.g., math tool, LLM tool).
- Uses a prompt template and LLMChain for flexible queries.
- Demonstrates multi-turn conversation and tool use.

**How to run:**
```bash
python agent_conversational.py
```

---

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ali3dev/LangChain.git
   cd LangChain
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your `.env` file:**
   ```
   GOOGLE_API_KEY=your-google-api-key-here
   ```

4. **Add your data:**
   - Place your text files (e.g., `odyssey.txt`) in the `RAG/books/` folder.

---

## 📦 Requirements

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [sentence-transformers](https://www.sbert.net/)
- [langchain-google-genai](https://github.com/langchain-ai/langchain-google-genai)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

See `requirements.txt` for the full list.

---

## 🤖 Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Google Gemini](https://ai.google.dev/gemini-api/docs/)

---

## 📄 License

This project is for educational and experimental purposes.

---
