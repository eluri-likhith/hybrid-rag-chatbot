# 📄 Hybrid RAG Document AI Assistant

A document question-answering application built using **Retrieval-Augmented Generation (RAG)**. Users can upload PDF or Word documents and ask questions about their contents. The system retrieves relevant text using hybrid search, reranks the retrieved chunks, and generates grounded answers using a Large Language Model (LLM).

## 🚀 Features

- **Document Upload:** Supports PDF and DOCX files.
- **Text Chunking:** Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Dense Retrieval:** Uses ChromaDB with sentence-transformer embeddings to retrieve semantically relevant content.
- **Sparse Retrieval:** Uses BM25 keyword-based search to find matching text.
- **Hybrid Search:** Combines dense and sparse results using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Reranks retrieved chunks to improve relevance.
- **LLM Answer Generation:** Uses Groq-hosted `openai/gpt-oss-20b` to generate concise answers based on retrieved context.
- **Source Information:** Displays document names, chunk identifiers, and page numbers when available.
- **Interactive UI:** Provides a simple Streamlit interface for uploading documents and asking questions.
- **Grounded Responses:** Instructs the LLM to acknowledge when the provided context does not contain the answer.

## 🏗️ Architecture

```text
Upload PDF / DOCX
       |
       v
Document Loading
       |
       v
Text Chunking
       |
       v
+---------------------+
|  Hybrid Retrieval   |
|                     |
|  ChromaDB Dense     |
|  BM25 Sparse        |
+---------------------+
       |
       v
Reciprocal Rank Fusion
       |
       v
Cross-Encoder Reranking
       |
       v
Top Relevant Chunks
       |
       v
Groq LLM Answer Generation
       |
       v
Answer + Source Information
       |
       v
Streamlit Interface
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM Provider | Groq |
| Language Model | `openai/gpt-oss-20b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Sparse Retrieval | BM25 (`rank_bm25`) |
| Rank Fusion | Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Document Processing | LangChain, PyPDF, Docx2txt |
| Environment Variables | python-dotenv |

## 📂 Project Structure

```text
hybrid-rag-document-assistant/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .gitignore
├── README.md
└── .env                 # Local only; never commit
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hybrid-rag-document-assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own Groq API key. Never commit `.env` or expose your API key publicly.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## 💬 How to Use

1. Launch the Streamlit application.
2. Upload a PDF or DOCX document.
3. Enter a question about the uploaded document.
4. Click **Get Answer**.
5. Review the generated answer and displayed source information.

## 🔍 How Hybrid Retrieval Works

### Dense Retrieval
Documents are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB. Dense retrieval finds passages with similar semantic meaning, even when they do not share the exact words in the query.

### Sparse Retrieval
BM25 ranks document chunks according to keyword relevance, helping retrieve passages that contain important terms from the question.

### Reciprocal Rank Fusion
RRF combines the ranked results from dense and sparse retrieval into a single ranking.

### Cross-Encoder Reranking
A cross-encoder scores the relationship between the user's query and each retrieved chunk. The highest-ranked chunks are passed to the LLM.

### Grounded Answer Generation
The retrieved chunks are provided as context to the Groq-hosted LLM. The prompt instructs the model to answer from that context and avoid inventing missing information.

## 🔐 Security Considerations

- Keep API keys in environment variables or a deployment secrets manager.
- Never commit `.env`, credentials, or private documents.
- Public deployments should use appropriate usage limits and monitoring.
- Uploaded documents may contain sensitive information; only upload content you are authorized to process.
- LLM responses can still be incorrect. Verify important answers against the original document.

## ⚠️ Current Limitations

- The application currently supports PDF and DOCX files.
- Scanned or image-only PDFs may require OCR before text can be retrieved.
- Retrieval quality depends on document quality, chunking, embeddings, and ranking.
- The LLM may occasionally produce inaccurate answers despite context-grounding instructions.
- The current implementation builds retrieval components for each question, which may increase response time for large documents.
- Retrieval evaluation metrics such as Recall@K, Precision@K, MRR, and NDCG are not currently exposed in the Streamlit interface.

## 🔮 Future Improvements

- Add retrieval evaluation using Recall@K, Precision@K, MRR, and NDCG.
- Measure end-to-end latency and individual pipeline stages.
- Add document caching and incremental indexing.
- Support multiple documents in a single knowledge base.
- Add automated tests, logging, and error monitoring.
- Improve deployment configuration and resource management.
- Add OCR support for scanned PDFs.
- Deploy the application using Streamlit Community Cloud.

## 👨‍💻 Project Purpose

This project demonstrates practical implementation of a Hybrid RAG pipeline, combining semantic retrieval, keyword retrieval, rank fusion, reranking, and LLM-based answer generation in an interactive document assistant.

## 📜 License

Unless a license is added to this repository, the reuse and redistribution permissions for this project are not explicitly granted. Add an appropriate open-source license if you want to permit reuse under defined terms.
