from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import torch
import warnings
import requests
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from lime.lime_text import LimeTextExplainer

TMDB_API_KEY = "8fbff5545ff86558e8b6fd669a4c8076"

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

app = FastAPI(title="CINEIQ AI Engine", description="Sentiment-Aware Recommendation API")

class MovieRequest(BaseModel):
    title: str

class Recommendation(BaseModel):
    title: str
    score: float
    lime_report: str

df = pd.read_csv('../data_clean/clean_masterdata.csv')
df = df.drop_duplicates(subset=['userId', 'movieId']).reset_index(drop=True)


df['user_cat'] = df['userId'].astype('category')


unique_movies = df[['title', 'overview', 'genres']].drop_duplicates(subset=['title']).reset_index(drop=True)
unique_movies['mix'] = unique_movies['overview'].fillna('') + " " + unique_movies['genres'].fillna('')


title_to_row = pd.Series(unique_movies.index, index=unique_movies['title']).to_dict()
df['movie_idx'] = df['title'].map(title_to_row)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(unique_movies['mix'])

sparse_matrix = csr_matrix((df['rating'], (df['movie_idx'], df['user_cat'].cat.codes)))
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(sparse_matrix)


indices = pd.Series(
    unique_movies.index,
    index=unique_movies['title'].str.lower().str.strip()
).to_dict()


sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])


def fetch_real_reviews(candidate_title):
    real_reviews = []
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={candidate_title}"
        search_response = requests.get(search_url, timeout=3).json()

        if search_response.get('results'):
            movie_id = search_response['results'][0]['id']
            reviews_url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}"
            reviews_response = requests.get(reviews_url, timeout=3).json()
            all_reviews = reviews_response.get('results', [])

            num_to_grab = min(5, len(all_reviews))
            if num_to_grab > 0:
                real_reviews = [r['content'] for r in random.sample(all_reviews, num_to_grab)]
    except Exception as e:
        print(f"  -> Skipping reviews for {candidate_title} (Network/API Error)")

    return real_reviews

def get_sentiment_score(text):
    if not text.strip(): return 0.0
    result = sentiment_analyzer(text[:512])[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']


def generate_lime_explanation(text):
    def predictor(texts):
        results = sentiment_analyzer(texts)
        probs = np.zeros((len(texts), 2))
        for i, res in enumerate(results):
            if res['label'] == 'POSITIVE':
                probs[i, 1] = res['score']
                probs[i, 0] = 1 - res['score']
            else:
                probs[i, 0] = res['score']
                probs[i, 1] = 1 - res['score']
        return probs

    exp = explainer.explain_instance(text, predictor, num_features=2, num_samples=50)
    keywords = [word for word, weight in exp.as_list() if weight > 0]
    return f"Recommended based on keywords: {', '.join(keywords)}." if keywords else "Recommended based on strong algorithmic matching."


@app.get("/")
def health_check():
    return {"status": "CINEIQ Engine is running!"}


@app.post("/recommend", response_model=list[Recommendation])
def get_recommendations(req: MovieRequest):
    clean_title = req.title.lower().strip()
    if clean_title not in indices:
        raise HTTPException(status_code=404, detail="Movie not found in database.")

    idx = indices[clean_title]
    content_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    collab_scores = cosine_similarity(latent_matrix[idx].reshape(1, -1), latent_matrix).flatten()
    hybrid_scores = (content_scores * 0.5) + (collab_scores * 0.5)

    sim_scores = sorted(list(enumerate(hybrid_scores)), key=lambda x: x[1], reverse=True)[1:16]
    final_output_data = []

    for i, (movie_idx, combined_score) in enumerate(sim_scores):
        candidate_title = unique_movies.iloc[movie_idx]['title']
        base_score = 15 - i

        real_reviews = fetch_real_reviews(candidate_title)
        real_reviews = [" ".join(r.split()[:350]) for r in real_reviews]
        sentiment_modifier = sum([get_sentiment_score(r) for r in real_reviews]) / len(
            real_reviews) if real_reviews else 0.0

        final_output_data.append({
            'score': base_score + (sentiment_modifier * 4.0),
            'title': candidate_title,
            'reviews_cache': real_reviews
        })

    top_3 = sorted(final_output_data, key=lambda x: x['score'], reverse=True)[:3]

    response_payload = []
    for movie in top_3:
        reviews = movie['reviews_cache']
        lime_report = "Recommended based on strong algorithmic matching."
        if reviews:
            review_scores = [(r, get_sentiment_score(r)) for r in reviews]
            review_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            lime_report = generate_lime_explanation(review_scores[0][0])

        response_payload.append(Recommendation(
            title=movie['title'],
            score=movie['score'],
            lime_report=lime_report
        ))

    return response_payload

