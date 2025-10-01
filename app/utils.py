"""
Utility functions for RAG Research Paper Discovery Tool
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import time
from typing import List, Dict, Tuple
import os
import scipy.stats as stats 

# Global variables for loaded resources
_model = None
_index = None
_df = None

# --- FILE PATH SETUP (Fix for FileNotFoundError) ---
# Define the base directory (the directory containing this utils.py file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct path logic: From 'app' folder, go up one (..) to project root, then down into 'data'
DATA_DIR = os.path.join(BASE_DIR, '..', 'data') 

def load_resources():
    """Load embedding model, FAISS index, and paper dataset using lazy loading."""
    global _model, _index, _df
    
    # Define the full file paths
    FAISS_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')
    PAPERS_PATH = os.path.join(DATA_DIR, 'arxiv_papers_clean.csv')
    
    # Initialize the model once
    if _model is None:
        try:
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"CRITICAL: Failed to load SentenceTransformer model: {e}")
            raise
    
    # Initialize the index once
    if _index is None:
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                f"FAISS index file not found at expected path: {FAISS_PATH}. "
                "Ensure you have created the index (e.g., ran notebook 03) and the file exists."
            )
        try:
            _index = faiss.read_index(FAISS_PATH)
        except Exception as e:
            raise IOError(f"Error loading FAISS index from {FAISS_PATH}. Check file integrity. Error: {e}")

    # Initialize the DataFrame once
    if _df is None:
        if not os.path.exists(PAPERS_PATH):
            raise FileNotFoundError(
                f"Papers CSV file not found at expected path: {PAPERS_PATH}. "
                "Ensure the data file exists."
            )
        _df = pd.read_csv(PAPERS_PATH)
    
    return _model, _index, _df

def retrieve_papers(query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Retrieve papers most relevant to the query using semantic search
    
    Args:
        query: User's search query
        top_k: Number of papers to retrieve
    
    Returns:
        DataFrame with retrieved papers and relevance scores
    """
    model, index, df = load_resources()
    
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    # Search in FAISS
    distances, indices = index.search(query_normalized.astype('float32'), top_k)
    
    # Get papers
    retrieved_papers = df.iloc[indices[0]].copy()
    retrieved_papers['relevance_score'] = distances[0]
    retrieved_papers['rank'] = range(1, len(retrieved_papers) + 1)
    
    return retrieved_papers

def summarize_text(
    text: str, 
    model_name: str = "facebook/bart-large-cnn",
    api_key: str = None,
    max_length: int = 130
) -> Tuple[str, float]:
    """
    Generate summary using Hugging Face Inference API
    
    Args:
        text: Text to summarize
        model_name: HuggingFace model identifier
        api_key: HuggingFace API key
        max_length: Maximum summary length
    
    Returns:
        Tuple of (summary_text, generation_time)
    """
    if not api_key or not api_key.startswith('hf_'):
        # Ensure error message starts with "Error:" for detection in app.py
        return "Error: Hugging Face API key not configured or invalid. Please check your settings.", 0
    
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # IMPROVEMENT (Issue 4): Increase truncation limit for better quality
    text_words = text.split()
    if len(text_words) > 750: 
        text = ' '.join(text_words[:750])
    
    # Special handling for T5 models
    if 't5' in model_name.lower():
        text = f"summarize: {text}"
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": max_length,
            "min_length": 30,
            "do_sample": False
        }
    }
    
    try:
        start_time = time.time()
        # Use a generous timeout for the API call
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60) 
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Handle different response formats
                summary = result[0].get('summary_text') or result[0].get('generated_text', '')
                return summary, elapsed_time
        
        # Handle model loading (503 Service Unavailable)
        if response.status_code == 503:
            return "Error: Model is loading on the Hugging Face server, please wait 30 seconds and try again...", elapsed_time
        
        # Return generic error from API
        return f"Error: Hugging Face API (Status {response.status_code}): {response.text[:100]}...", elapsed_time
    
    except requests.exceptions.Timeout:
        return "Error: Hugging Face API request timed out (60 seconds).", 0
    except Exception as e:
        return f"Error: during API call: {str(e)}", 0


def get_model_configs() -> Dict:
    """
    Get configurations for A/B testing models.
    (Fixes Issue 2: Naming Mismatch)
    """
    return {
        'model_a': {
            'name': 'facebook/bart-large-cnn',
            'display_name': 'Model A (BART)',
            'description': 'BART-large fine-tuned on CNN/DailyMail (High Quality)'
        },
        'model_b': {
            'name': 'sshleifer/distilbart-cnn-6-6', # Changed to a DistilBART variant
            'display_name': 'Model B (DistilBART)', # Changed display name
            'description': 'DistilBART for faster, smaller model summarization'
        }
    }

def log_ab_test_result(
    query: str,
    paper_title: str,
    model_chosen: str,
    summary_a: str,
    summary_b: str,
    log_file: str = 'ab_test_logs.csv' # Note: Updated path handling below
):
    """
    Log user's A/B test preference
    """
    LOG_FILE_PATH = os.path.join(DATA_DIR, log_file)
    
    log_entry = {
        'timestamp': pd.Timestamp.now(),
        'query': query,
        'paper_title': paper_title,
        'model_chosen': model_chosen,
        'summary_a': summary_a,
        'summary_b': summary_b
    }
    
    # Append to CSV
    df_log = pd.DataFrame([log_entry])
    
    if os.path.exists(LOG_FILE_PATH):
        df_log.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE_PATH, index=False)

def analyze_ab_test_logs(log_file: str = 'ab_test_logs.csv') -> Dict:
    """
    Analyze A/B test results from logs using a safety check for low data volume.
    """
    LOG_FILE_PATH = os.path.join(DATA_DIR, log_file)
    
    if not os.path.exists(LOG_FILE_PATH):
        return {'status': 'warning', 'message': 'Log file not found. Run A/B tests first.'}
    
    df = pd.read_csv(LOG_FILE_PATH)
    
    if 'model_chosen' not in df.columns:
        return {'status': 'error', 'message': 'Log file is corrupted (missing "model_chosen" column).'}
    
    # Calculate preference rates
    total_votes = len(df)
    model_a_votes = (df['model_chosen'] == 'A').sum()
    model_b_votes = (df['model_chosen'] == 'B').sum()
    equal_votes = (df['model_chosen'] == 'Equal').sum()
    
    results = {
        'total_votes': total_votes,
        'model_a_votes': model_a_votes,
        'model_b_votes': model_b_votes,
        'equal_votes': equal_votes,
        'model_a_rate': model_a_votes / total_votes if total_votes > 0 else 0,
        'model_b_rate': model_b_votes / total_votes if total_votes > 0 else 0,
        'status': 'success'
    }
    
    # Chi-square assumption check: require at least 30 total votes and 5 votes per model
    MIN_VOTES_FOR_STATS = 30 
    # For chi-square test, we only compare A vs B (excluding "Equal" votes)
    ab_only_votes = model_a_votes + model_b_votes

    if total_votes < MIN_VOTES_FOR_STATS or model_a_votes < 5 or model_b_votes < 5:
        # Insufficient data for reliable Chi-Square test. Return a warning instead of error.
        results['status'] = 'warning'
        results['message'] = f'Insufficient data for statistical analysis (need at least {MIN_VOTES_FOR_STATS} votes and > 5 votes per model).'
        return results

    # Chi-square test 
    try:
        from scipy.stats import chisquare
        observed = [model_a_votes, model_b_votes]
       
        # Expected is 50/50 split of ONLY the A+B votes (not total_votes)
        expected = [ab_only_votes/2, ab_only_votes/2]
        
        # This test compares preference against a null hypothesis of equal preference (50/50)
        chi2, p_value = chisquare(observed, expected) 
        
        results['chi2_statistic'] = chi2
        results['p_value'] = p_value
        results['significant'] = p_value < 0.05
    except Exception as e:
        results['status'] = 'error'
        results['message'] = f'Unexpected error during Chi-Square test: {e}'

    return results

def format_paper_card(paper: pd.Series, show_abstract: bool = True) -> str:
    """
    Format paper information for display
    
    Args:
        paper: Paper data as pandas Series
        show_abstract: Whether to include abstract
    
    Returns:
        Formatted string
    """
    card = f"**{paper['title']}**\n\n"
    card += f"*Authors*: {paper['authors'][:100]}...\n\n"
    card += f"*Categories*: {paper['categories']}\n\n"
    card += f"*Published*: {paper['published_date']}\n\n"
    
    # Changed from 'abstract_clean' to 'abstract' assuming it's the right column
    if show_abstract and 'abstract' in paper: 
        abstract_preview = paper['abstract'][:300]
        card += f"*Abstract*: {abstract_preview}...\n\n"
    
    if 'pdf_url' in paper:
        card += f"[📄 View PDF]({paper['pdf_url']})\n\n"
    
    if 'relevance_score' in paper:
        card += f"*Relevance Score*: {paper['relevance_score']:.4f}"
    
    return card

def get_category_stats(df: pd.DataFrame = None) -> Dict:
    """Get statistics about paper categories"""
    if df is None:
        _, _, df = load_resources()
    
    all_categories = df['categories'].str.split(', ').explode()
    category_counts = all_categories.value_counts()
    
    return {
        'total_categories': len(category_counts),
        'top_categories': category_counts.head(10).to_dict(),
        'papers_per_category': category_counts.describe().to_dict()
    }

def get_dataset_stats(df: pd.DataFrame = None) -> Dict:
    """Get general dataset statistics"""
    if df is None:
        _, _, df = load_resources()
    
    # Ensure 'abstract_clean' is used if it exists, otherwise use 'abstract'
    abstract_col = 'abstract_clean' if 'abstract_clean' in df.columns else 'abstract'

    if abstract_col in df.columns:
        df['abstract_word_count'] = df[abstract_col].str.split().str.len()
    else:
        df['abstract_word_count'] = 0 # Default if column is missing

    df['published_date'] = pd.to_datetime(df['published_date'])
    
    return {
        'total_papers': len(df),
        'date_range': {
            'earliest': df['published_date'].min().strftime('%Y-%m-%d'),
            'latest': df['published_date'].max().strftime('%Y-%m-%d')
        },
        'abstract_stats': {
            'mean_words': df['abstract_word_count'].mean(),
            'min_words': df['abstract_word_count'].min(),
            'max_words': df['abstract_word_count'].max()
        },
        'categories': get_category_stats(df)
    }