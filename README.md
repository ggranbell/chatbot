
# 🤖 Mola Chatbot

  *A high-performance, local-first RAG ecosystem powered by LangChain and Ollama.*

  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-Inference-ED1C24?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

<br />

🚀 About Mola Chatbot
<hr>

**Mola Chatbot** is a professional-grade Retrieval-Augmented Generation (RAG) system designed for privacy-conscious environments. It enables users to chat with their own document collections (PDF, DOCX, XLSX) using local AI models, ensuring that sensitive data never leaves the local machine.

### Key Capabilities

- 🧠 **Local Intelligence**: Leverages Ollama for state-of-the-art inference (Qwen 3.5, Llama 3, etc.).
- 📂 **Multi-format Ingestion**: Seamlessly handles PDF, Word, and Excel documents with OCR support.
- 🔍 **Hybrid Search**: Combines BM25 lexical search with LanceDB vector embeddings for maximum precision.
- ⚡ **Reranking**: Integrated Sentence Transformers (Cross-Encoders) to ensure only the most relevant context reaches the LLM.
- 🌐 **Clean API**: Fully functional FastAPI backend with a reactive vanilla frontend.

<br />

🛠️ Tech Stack & Tools
<hr>

### AI, ML & Deep Learning

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![LanceDB](https://img.shields.io/badge/LanceDB-000000?style=for-the-badge&logo=lancedb&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FDE047?style=for-the-badge&logo=huggingface&logoColor=black)

### Backend Framework

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499BEA?style=for-the-badge&logo=python&logoColor=white)

### Preprocessing & OCR

![PyMuPDF](https://img.shields.io/badge/PyMuPDF-FF1493?style=for-the-badge)
![Tesseract](https://img.shields.io/badge/Tesseract--OCR-4B0082?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Frontend & UI

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

<br />

💻 System Requirements
<hr>

To run Mola Chatbot effectively with the default local models:

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | 4 Cores (Modern) | 8+ Cores |
| **RAM** | 8 GB | 16 GB - 32 GB |
| **GPU** | Optional (CPU only supported) | 8 GB+ VRAM (NVIDIA RTX) |
| **Storage** | 10 GB Free | SSD Recommended |
| **OS** | Windows / Linux / macOS | Linux (Ubuntu 22.04+) |

> [!IMPORTANT]
> A GPU is strongly recommended for the `SentenceTransformers` reranker and the 9B parameter LLM to achieve interactive response times.

<br />

⚙️ Getting Started
<hr>

### 1. Prerequisites

Ensure you have [Ollama](https://ollama.com/) installed and running:

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

### 2. Installation

Clone the repository and install dependencies:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Running the Server

Start the FastAPI server:

```bash
python app.py
```

The application will be available at `http://localhost:8080`.

<br />

🧩 RAG Pipeline Architecture
<hr>

Mola uses a **two-stage hybrid retrieval** architecture:

1. **Stage 1: Retrieval**
    - **Vector Search**: Semantic retrieval using `LanceDB` and `nomic-embed-text`.
    - **Lexical Search**: Keyword-based retrieval using `BM25`.
    - Both signals are combined via an `EnsembleRetriever`.
2. **Stage 2: Reranking**
    - The top candidates are passed through a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`).
    - This validates the actual relevance of the context before passing it to the final LLM.

<br />

---
<div align="center">
  Developed by <b>Granbell</b> • Built for <b>Advanced Agentic Coding</b>
</div>
