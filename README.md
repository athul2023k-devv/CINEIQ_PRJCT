# CINEIQ: Sentiment-Aware Hybrid Recommendation Engine

# Overview
CINEIQ is a full-stack Machine Learning microservice designed to generate highly accurate movie recommendations. It moves beyond basic collaborative filtering by combining a hybrid mathematical model (TF-IDF and SVD) with real-time Natural Language Processing (NLP). By scraping live reviews and processing them through a HuggingFace DistilBERT model, the engine dynamically adjusts algorithmic recommendations based on current audience sentiment.

## Core Architecture
* **Hybrid Recommender System:** Combines Content-Based Filtering (utilizing TF-IDF vectorization on plot and genre data) with Collaborative Filtering (utilizing TruncatedSVD on user rating matrices).
* **Real-Time Sentiment Analysis:** Integrates HuggingFace DistilBERT to process live internet reviews, applying a mathematical penalty to mathematically similar movies that exhibit poor modern sentiment.
* **Explainable AI (XAI):** Implements LIME (Local Interpretable Model-agnostic Explanations) to output human-readable justifications for algorithmic decisions.
* **Microservice Architecture:** Strictly decoupled environment featuring a FastAPI backend for matrix computation and a Streamlit frontend for state management and UI rendering.
* **Interactive Analytics:** Features a user taste profiler leveraging Plotly for advanced data visualization (Radar charts, histograms, and affinity mapping).

## Technology Stack
* **Backend:** FastAPI, Uvicorn, Python Requests
* **Frontend:** Streamlit, Plotly
* **Machine Learning:** Scikit-Learn, HuggingFace Transformers, PyTorch, LIME
* **Data Processing:** Pandas, Regex, AST

## Repository Structure
```text
CINEIQ_Project/
├── data_clean/
│   └── clean_masterdata.csv      # Must be downloaded manually (see instructions)
├── api.py                        # FastAPI backend server
├── app.py                        # Streamlit frontend application
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```
## Installation setup
* Due to GitHub file restrictions you can download the masterdata set (cleaned) through the following
* ## google drive link: https://drive.google.com/drive/folders/14_Y-H7yf3C0O8DvWgfp8LP4yQSWOiodH 
* Ensure directory path: data_clean/clean_masterdata.csv

* The repo contains the requirements(libraries) as well (requirements.txt

## Execution
* Backend initialisation (uvicorn api:app)
* Frontend initialisation (streamlit run app.py)
* Above cmnds in bash 
## Note
* Since the abovve code involves an api call to tmdb's server have to use vpn(network blocks the site)
