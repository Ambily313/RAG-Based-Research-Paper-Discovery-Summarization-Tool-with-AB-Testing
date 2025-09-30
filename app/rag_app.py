"""
RAG Research Paper Discovery & Summarization Tool
Interactive Streamlit Application with A/B Testing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    retrieve_papers,
    summarize_text,
    get_model_configs,
    log_ab_test_result,
    analyze_ab_test_logs,
    format_paper_card,
    get_dataset_stats
)
import os
from dotenv import load_dotenv
import traceback # Added for better error logging (optional but useful)

# Page config
st.set_page_config(
    page_title="RAG Research Paper Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# --- Helper Function for Summary Display (Fixes Issue 1 & 3) ---
def display_summary(summary_text, gen_time=None):
    """Checks if the summary is an error message and displays it accordingly."""
    # Check if the text is an error message returned from the utility function
    is_error = summary_text.lower().startswith("error:")
    
    if is_error:
        # Display API/Network errors in a red box
        st.error(summary_text)
    else:
        # Display successful summaries with the custom style
        st.markdown(f"<div class='summary-box'>{summary_text}</div>", 
                    unsafe_allow_html=True)
    
    if gen_time is not None:
        st.caption(f"⏱️ Generation time: {gen_time:.2f}s")
    if not is_error:
        st.caption(f"📏 Length: {len(summary_text.split())} words")
# -----------------------------------------------------------------


# Custom CSS with elegant green button
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .summary-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Elegant green button styling */
    .stButton > button[kind="primary"] {
        background-color: #10B981 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================

st.sidebar.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=RAG+Research+Tool", use_column_width=True)

# Changed from "Select Page" to "Select What You Want to Perform"
st.sidebar.markdown("### 🎯 Choose An Action to Perform")

page = st.sidebar.radio(
    "",  # Removed label since it's now in markdown above
    ["🔍 Search & Summarize", "🧪 A/B Testing", "📊 Analytics", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Search parameters
with st.sidebar.expander("🔧 Search Parameters"):
    top_k = st.slider("Number of papers to retrieve", 1, 10, 5)
    summary_length = st.slider("Summary max length (words)", 100, 350, 250)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")

# Load and display dataset stats
try:
    # Ensure this runs only once when the app starts
    if 'stats' not in st.session_state:
        st.session_state['stats'] = get_dataset_stats()
        
    stats = st.session_state['stats']
    st.sidebar.metric("Total Papers", stats['total_papers'])
    st.sidebar.metric("Categories", stats['categories']['total_categories'])
except Exception as e:
    st.sidebar.warning(f"Could not load stats: {str(e)}")

# ================================
# PAGE 1: SEARCH & SUMMARIZE
# ================================

if page == "🔍 Search & Summarize":
    st.markdown("<h1 class='main-header'>📚 Research Paper Discovery</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the RAG-based research paper discovery tool! Enter your research question 
    or topic below to find relevant papers and generate AI-powered summaries.
    """)
    
    # Search interface
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your research query:",
            placeholder="e.g., attention mechanisms in transformers"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Use session state to preserve search results across re-runs
    if search_button and query:
        with st.spinner("🔎 Searching papers..."):
            try:
                # Retrieve papers
                papers = retrieve_papers(query, top_k=top_k)
                st.session_state['search_papers'] = papers
                st.session_state['search_query'] = query
                
                st.success(f"✅ Found {len(papers)} relevant papers for: **{query}**")
            
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
                st.info("Make sure all data files (faiss_index.bin, arxiv_papers_clean.csv) are in the `../data/` directory")
    
    # Display results if they exist in session state
    if 'search_papers' in st.session_state:
        papers = st.session_state['search_papers']
        
        # Display results
        for idx, row in papers.iterrows():
            with st.container():
                st.markdown(f"### {row['rank']}. {row['title']}")
                
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"**Authors:** {row['authors'][:100]}...")
                    st.markdown(f"**Categories:** {row['categories']}")
                    st.markdown(f"**Published:** {row['published_date']}")
                    st.markdown(f"**Relevance Score:** {row['relevance_score']:.4f}")
                
                with col_actions:
                    pdf_url = row.get('pdf_url')
                    if pdf_url and st.button(f"📄 View PDF", key=f"pdf_{idx}"):
                        st.markdown(f"[Open PDF]({pdf_url})")
                
                # Abstract
                with st.expander("📖 View Abstract"):
                    abstract_text = row.get('abstract_clean', row.get('abstract', 'No abstract available'))
                    st.write(abstract_text)
                
                # Generate summary
                # Use a specific key for the summary button to trigger only this block on click
                if st.button(f"✨ Generate Summary", key=f"sum_{idx}"):
                    if not HF_API_KEY:
                        st.error("❌ Summarization service is not configured. Please set the HUGGINGFACE_API_KEY environment variable.")
                    else:
                        with st.spinner("Generating summary..."):
                            abstract_text_for_sum = row.get('abstract_clean', row.get('abstract', ''))
                            
                            try:
                                summary, gen_time = summarize_text(
                                    abstract_text_for_sum,
                                    model_name="facebook/bart-large-cnn",
                                    api_key=HF_API_KEY, 
                                    max_length=summary_length
                                )
                                
                                st.markdown("#### 📝 AI-Generated Summary")
                                # Use the helper function to handle success/error display
                                display_summary(summary, gen_time)
                            
                            except Exception as e:
                                # Catch any unexpected exceptions during the summarization process
                                st.error(f"An unexpected error occurred during summarization. Check console for details. Error: {str(e)}")
                                # Optional: print full traceback to the console for debugging
                                # print(traceback.format_exc())
                
                st.markdown("---")

    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "deep learning for computer vision",
            "natural language processing transformers",
            "reinforcement learning robotics",
            "graph neural networks",
            "few-shot learning",
            "attention mechanisms"
        ]
        st.markdown("Try these example queries:")
        for ex in examples:
            st.markdown(f"- {ex}")

# ================================
# PAGE 2: A/B TESTING
# ================================

elif page == "🧪 A/B Testing":
    st.markdown("<h1 class='main-header'>🧪 Model A/B Testing</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Help us determine which summarization model performs better! 
    Search for papers, compare summaries from two different models, and vote for your preference.
    """)
    
    # Get model configs (Uses updated names from utils.py, addressing Issue 2)
    models = get_model_configs()
    
    # Display model info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🤖 {models['model_a']['display_name']}")
        st.info(f"""
        **Name:** {models['model_a']['display_name']}  
        **Description:** {models['model_a']['description']}
        """)
    with col2:
        st.markdown(f"### 🤖 {models['model_b']['display_name']}")
        st.info(f"""
        **Name:** {models['model_b']['display_name']}  
        **Description:** {models['model_b']['description']}
        """)
    
    st.markdown("---")
    
    # Search interface
    query_ab = st.text_input("Search for papers:", placeholder="e.g., neural machine translation")
    
    if st.button("🔍 Search for A/B Test", type="primary"):
        if query_ab:
            with st.spinner("Searching papers..."):
                papers_ab = retrieve_papers(query_ab, top_k=3)
                st.session_state['ab_papers'] = papers_ab
                st.session_state['ab_query'] = query_ab
                # Clear summaries when performing a new search
                if 'summary_a' in st.session_state: del st.session_state['summary_a']
                if 'summary_b' in st.session_state: del st.session_state['summary_b']
        else:
            st.warning("⚠️ Please enter a search query")
    
    # Display papers for A/B testing
    if 'ab_papers' in st.session_state:
        st.markdown("### 📄 Select a Paper to Compare Summaries")
        
        papers_ab = st.session_state['ab_papers']
        
        selected_paper_idx = st.selectbox(
            "Choose a paper:",
            range(len(papers_ab)),
            format_func=lambda i: f"{papers_ab.iloc[i]['title'][:80]}..."
        )
        
        selected_paper = papers_ab.iloc[selected_paper_idx]
        
        # Display paper info
        with st.expander("📖 Paper Details", expanded=True):
            st.markdown(format_paper_card(selected_paper))
        
        # Generate summaries button
        if st.button("✨ Generate Summaries from Both Models"):
            if not HF_API_KEY:
                st.error("❌ Summarization service is not configured. Please set the HUGGINGFACE_API_KEY environment variable.")
            else:
                with st.spinner("Generating summaries from both models..."):
                    abstract_text = selected_paper.get('abstract_clean', selected_paper.get('abstract', ''))
                    
                    try:
                        # Generate with Model A
                        summary_a, time_a = summarize_text(
                            abstract_text,
                            model_name=models['model_a']['name'],
                            api_key=HF_API_KEY,
                            max_length=summary_length
                        )
                        
                        # Generate with Model B
                        summary_b, time_b = summarize_text(
                            abstract_text,
                            model_name=models['model_b']['name'],
                            api_key=HF_API_KEY,
                            max_length=summary_length
                        )
                        
                        # Store in session state
                        st.session_state['summary_a'] = summary_a
                        st.session_state['summary_b'] = summary_b
                        st.session_state['time_a'] = time_a
                        st.session_state['time_b'] = time_b
                        st.session_state['current_paper'] = selected_paper
                    
                    except Exception as e:
                        st.error(f"An unexpected error occurred during A/B summarization. Error: {str(e)}")
        
        # Display summaries and voting
        if 'summary_a' in st.session_state and 'summary_b' in st.session_state:
            st.markdown("### 📊 Compare Summaries")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Summary A ({models['model_a']['display_name'].split('(')[-1].replace(')','')})")
                # Use helper function for robust display (Issue 3 Fix)
                display_summary(st.session_state['summary_a'], st.session_state['time_a'])

            with col2:
                st.markdown(f"#### Summary B ({models['model_b']['display_name'].split('(')[-1].replace(')','')})")
                # Use helper function for robust display (Issue 3 Fix)
                display_summary(st.session_state['summary_b'], st.session_state['time_b'])
            
            # Voting
            st.markdown("### 🗳️ Which summary is better?")
            
            col_vote1, col_vote2, col_vote3 = st.columns([1, 1, 1])
            
            # Ensure voting buttons are only displayed if both summaries are NOT errors
            is_vote_enabled = (not st.session_state['summary_a'].lower().startswith("error:")) and \
                              (not st.session_state['summary_b'].lower().startswith("error:"))
            
            if is_vote_enabled:
                with col_vote1:
                    if st.button("👍 Summary A is Better", use_container_width=True):
                        log_ab_test_result(
                            query=st.session_state['ab_query'],
                            paper_title=st.session_state['current_paper']['title'],
                            model_chosen='A',
                            summary_a=st.session_state['summary_a'],
                            summary_b=st.session_state['summary_b']
                        )
                        st.success("✅ Thank you! Your vote has been recorded.")
                
                with col_vote2:
                    if st.button("👍 Summary B is Better", use_container_width=True):
                        log_ab_test_result(
                            query=st.session_state['ab_query'],
                            paper_title=st.session_state['current_paper']['title'],
                            model_chosen='B',
                            summary_a=st.session_state['summary_a'],
                            summary_b=st.session_state['summary_b']
                        )
                        st.success("✅ Thank you! Your vote has been recorded.")
                
                with col_vote3:
                    if st.button("🤷 About Equal", use_container_width=True):
                        log_ab_test_result(
                            query=st.session_state['ab_query'],
                            paper_title=st.session_state['current_paper']['title'],
                            model_chosen='Equal',
                            summary_a=st.session_state['summary_a'],
                            summary_b=st.session_state['summary_b']
                        )
                        st.success("✅ Thank you! Your vote has been recorded.")
            else:
                 st.warning("Voting disabled: One or both summaries failed to generate. Please check your API key or try again later.")


# ================================
# PAGE 3: ANALYTICS
# ================================

elif page == "📊 Analytics":
    st.markdown("<h1 class='main-header'>📊 A/B Test Analytics</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    View the results of the A/B testing experiment comparing different summarization models.
    """)
    
    # Analyze logs
    try:
        results = analyze_ab_test_logs()
        
        if results.get('status') in ['warning', 'error']:
            st.info(f"ℹ️ {results.get('message', 'No test data available')}")
            st.markdown("""
            **No test data available yet.** Go to the A/B Testing page to start collecting data!
            """)
        else:
            # Display metrics
            st.markdown("### 📈 Overall Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Votes", results['total_votes'])
            with col2:
                st.metric("Model A Preference", f"{results['model_a_rate']*100:.1f}%")
            with col3:
                st.metric("Model B Preference", f"{results['model_b_rate']*100:.1f}%")
            
            # Visualizations
            st.markdown("### 📊 Visualizations")
            
            # Pie chart
            equal_votes = results['total_votes'] - results['model_a_votes'] - results['model_b_votes']
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Model A', 'Model B', 'Equal'],
                values=[results['model_a_votes'], results['model_b_votes'], equal_votes],
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c']
            )])
            fig_pie.update_layout(title="User Preference Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Statistical significance
            if 'significant' in results:
                st.markdown("### 🔬 Statistical Analysis")
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.metric("Chi-Square Statistic", f"{results['chi2_statistic']:.4f}")
                with col_stat2:
                    st.metric("P-Value", f"{results['p_value']:.4f}")
                
                if results['significant']:
                    winner = "Model A" if results['model_a_votes'] > results['model_b_votes'] else "Model B"
                    st.success(f"✅ **Statistically Significant Result**: {winner} is preferred (p < 0.05)")
                else:
                    st.info("ℹ️ No statistically significant difference detected (p >= 0.05)")
            else:
                st.info(f"ℹ️ {results.get('message', 'Need more votes for statistical analysis')}")
            
            # Show recent logs
            st.markdown("### 📋 Recent Test Logs")
            
            log_path = os.path.join('..', 'data', 'ab_test_logs.csv')
            if os.path.exists(log_path):
                df_logs = pd.read_csv(log_path)
                st.dataframe(
                    df_logs[['timestamp', 'query', 'paper_title', 'model_chosen']].tail(10),
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

# ================================
# PAGE 4: ABOUT
# ================================

elif page == "ℹ️ About":
    st.markdown("<h1 class='main-header'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This is a **Retrieval-Augmented Generation (RAG)** system for discovering and summarizing 
    academic research papers. The project demonstrates:
    
    - 📚 **Semantic Search**: Find papers by meaning, not just keywords
    - 🤖 **AI Summarization**: Generate concise summaries using LLMs
    - 🧪 **A/B Testing**: Compare model performance systematically
    - 📊 **Data-Driven Insights**: Statistical analysis of user preferences
    
    ## 🛠️ Technical Stack
    
    - **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
    - **Vector Database**: FAISS for fast similarity search
    - **LLMs**: HuggingFace Inference API (BART, DistilBART)
    - **Frontend**: Streamlit
    - **Data Source**: ArXiv API
    
    ## 📁 Project Structure
    
    ```
    rag-research-papers/
    ├── notebooks/            # Jupyter notebooks for analysis
    │   ├── 01_data_collection.ipynb
    │   ├── 02_eda_preprocessing.ipynb
    │   ├── 03_embeddings_faiss.ipynb
    │   ├── 04_rag_prototype.ipynb
    │   └── 05_ab_testing_analysis.ipynb
    ├── app/                # Streamlit application
    │   ├── rag_app.py
    │   └── utils.py
    └── data/               # Datasets and artifacts
        ├── arxiv_papers.csv
        ├── embeddings.npy
        ├── faiss_index.bin
        └── ab_test_logs.csv
    ```
    
    ## 🎓 Key Learnings
    
    This project demonstrates a **data scientist's approach** to building ML applications:
    
    1. **Start with exploration** (notebooks) before building the app
    2. **Validate data quality** through EDA
    3. **Experiment with models** in isolation
    4. **Measure performance** with quantitative metrics
    5. **Deploy iteratively** with user feedback loops
    
    ## 📖 How to Use
    
    1. **Search & Summarize**: Enter a research query to find relevant papers
    2. **A/B Testing**: Compare summaries from different models and vote
    3. **Analytics**: View aggregated results and statistical analysis
    
    ---
    
    **Version**: 1.0.4 (A/B & Summarization Fix)
    **Last Updated**: September 2025
    """)
    
    # Dataset statistics
    try:
        stats = st.session_state['stats']
        
        st.markdown("## 📊 Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", stats['total_papers'])
        with col2:
            st.metric("Unique Categories", stats['categories']['total_categories'])
        with col3:
            st.metric("Avg Abstract Length", f"{stats['abstract_stats']['mean_words']:.0f} words")
        
        st.markdown(f"""
        **Date Range**: {stats['date_range']['earliest']} to {stats['date_range']['latest']}
        """)
        
        # Top categories
        st.markdown("### 🏷️ Top Categories")
        top_cats = pd.DataFrame.from_dict(
            stats['categories']['top_categories'], 
            orient='index', 
            columns=['Count']
        )
        st.bar_chart(top_cats)
        
    except Exception as e:
        st.warning(f"Could not load dataset stats: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>📚 RAG Research Paper Discovery Tool | Built with Streamlit & HuggingFace</p>
    <p>Data Scientist Portfolio Project</p>
</div>
""", unsafe_allow_html=True)