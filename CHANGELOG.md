# 📜 CHANGELOG

## Data Engineering Design - v2 (2025-09-30)

### Old Design
- Extracted: Meta data and Abstract only.
- Stored: Metadata CSV file.
- Limitations: Limited semantic context → summaries were shallow.

### New Design
- Extracting: Full-text PDF content.
- Storage: JSON format .
- Benefits: Richer semantic search, improved summarization quality, better coverage of research papers.
 

---

## Version 2.1 Enhancements (2025-10-04)

###  1. Enhanced Search Capabilities
**Old:**  
Semantic search only (based on sentence-transformer embeddings).  

**New:**  
Added **keyword-based search** alongside **semantic search** for hybrid evaluation.  


**Benefit:**  
Provides flexibility in retrieval — combining semantic understanding with exact keyword matching improves search relevance and diversity.

---

### 🗂️ 2. Data Folder Reorganization
**Old:**  
All datasets and analysis outputs were stored directly in the `/data/` folder.  

**New:**  
Introduced structured subfolders for better modularity and data management:
**Benefit:** 
Improves organization, reproducibility, and scalability for future dataset additions.

```plaintext
data/
├── raw/                # Raw extracted CSV, JSON, and PDFs
├── processed/          # Cleaned data and embeddings
├── analysis/           # Reports, plots, and A/B test results
└── index/              # FAISS index, metadata, and vector files


