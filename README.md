# 🕷️ Conversational Q\&A with Local LLM + FAISS

This project demonstrates how to build a **Conversational Retrieval-Augmented Generation (RAG)** system using:

* A **local language model** (e.g., GPT-2),
* **FAISS** for fast vector-based similarity search,
* **LangChain** to manage the retrieval and conversation logic.

The system can answer questions about a text file (e.g., `spiderman.txt`) in a conversational format, remembering previous questions using chat history.

---

## 📂 Project Structure

```
.
├── spiderman.txt           # Input file with content to ask questions about
├── QnA_ChatBot.py                 # Python script with the full pipeline
└── README.md               # You're reading this!
```

---

## 🔧 Setup Instructions

### 1. Clone or download the repo

```bash
git clone https://github.com/soubhik9/LLM_UseCases
cd qa-chat-llm-faiss
```

### 2. Install dependencies

Make sure you’re using Python 3.8+ and install the required libraries:

```bash
pip install torch faiss-cpu transformers langchain sentence-transformers
```

> If you're using a GPU, install `faiss-gpu` instead of `faiss-cpu`.

---

## 🚀 How to Run

1. Put your document (e.g., `spiderman.txt`) in the same folder.
2. Run the script:

```bash
python main.py
```

It will:

* Load the document
* Embed it using SentenceTransformers
* Create a FAISS vector index
* Load GPT-2 for local text generation
* Start a QA session with memory of previous questions

---

## 🧠 What It Does

This app lets you ask questions like:

```txt
Q: What is the story about?
A: Peter Parker, a high school student bitten by a radioactive spider...
```

And it remembers previous questions in the same session, so you can ask:

```txt
Q: What powers did he gain?
A: He gained super strength, agility, wall-crawling, and a sixth sense...
```

---

## 🧱 Stack Used

* **LangChain** – Orchestration framework
* **FAISS** – Vector similarity search
* **SentenceTransformers** – For generating dense embeddings
* **Transformers (Hugging Face)** – GPT-2 for local LLM inference

---

## 🔀 Optional Improvements

* Swap GPT-2 with a larger model (`gpt2-large`, `llama`, etc.)
* Add a Streamlit UI for interactive chat
* Load PDFs instead of plain text
* Persist the FAISS index and chat memory

---

## 📜 License

MIT — free to use for personal or commercial projects.
