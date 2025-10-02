
# 📚 RAG-Based Research Paper Discovery & Summarization Tool with A/B Testing

A data science project demonstrating end-to-end ML pipeline development with **Retrieval-Augmented Generation (RAG)**, vector databases, and **A/B testing**.

---

##  Project Overview
This project builds a **semantic search and summarization system** for academic papers, showcasing:

- ✅ End-to-end data pipeline (collection → preprocessing → embeddings → retrieval)  
- ✅ RAG implementation with vector databases  
- ✅ Model evaluation through A/B testing  
- ✅ Interactive demo with Streamlit  

---
##  Key Features

- **Semantic Search** → Find papers by meaning using Sentence-BERT embeddings  
- **FAISS Integration** → Fast similarity search with minimal average latency  
- **AI Summarization** → Generate concise summaries using HuggingFace LLMs  
- **A/B Testing** → Compare **BART** vs **DistilBART** summarization models
  - Uses user preference logs + statistical analysis (Chi-square, t-tests)
  - Results help determine the best summarizer for academic papers
- **Interactive UI** → Streamlit app for live demos and user studies  


##  Tech Stack

| Component        | Technology                                   |
|------------------|-----------------------------------------------|
| **Language**     | Python 3.8+                                  |
| **Embeddings**   | Sentence-Transformers (`all-MiniLM-L6-v2`)   |
| **Vector DB**    | FAISS                                        |
| **LLMs**         | HuggingFace API (BART, DistilBART)                   |
| **Frontend**     | Streamlit                                    |
| **Data Source**  | ArXiv API                                    |
| **Analysis**     | Pandas, NumPy, SciPy, Matplotlib             |

---
##  Getting Started

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/RAG-Based-Research-Paper-Discovery-Summarization-Tool-with-AB-Testing.git
   cd RAG-Based-Research-Paper-Discovery-Summarization-Tool-with-AB-Testing
2. **Create a virtual environment & install dependencies**
   ```bash
   pip install -r requirements.txts
3. **Set up API keys**
   ```bash
   Copy .env.example → .env and add your HuggingFace / ArXiv credentials.

3. **Run the Streamlit app**
   ```bash
   streamlit run app/rag_app.py

## 📁 Project Structure

```plaintext

rag-research-papers/
│
├── notebooks/                          # ML workflow notebooks
│   ├── 01_data_collection.ipynb        # Scrape ArXiv papers
│   ├── 02_eda_preprocessing.ipynb      # Exploratory analysis & cleaning
│   ├── 03_embeddings_faiss.ipynb       # Generate embeddings & build FAISS index
│   ├── 04_rag_prototype.ipynb          # RAG prototype system
│   └── 05_ab_testing_analysis.ipynb    # A/B testing & statistical analysis
│
├── app/                                # Application code
│   ├── rag_app.py                      # Streamlit app
│   └── utils.py                        # Helper functions
│
├── data/                               # Data & artifacts
│   ├── arxiv_papers.csv                # Raw paper data
│   ├── arxiv_papers_clean.csv          # Preprocessed paper data
│   ├── embeddings.npy                  # Paper embeddings
│   ├── faiss_index.bin                 # FAISS vector index
│   └── ab_test_logs.csv                # User preference logs
│
├── docs/                               # Documentation
│   ├── Technical_Architecture.png      # System architecture diagram   
│   
├── requirements.txt                    # Python dependencies
├── .env                                # API keys template
├── CHANGELOG.md                        # Project changelog
└── README.md                           # Project documentation
