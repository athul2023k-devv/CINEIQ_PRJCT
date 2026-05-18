import pandas as pd
import ast

credits = pd.read_csv('../data_raw/TMDB_Meta/credits.csv')
movies = pd.read_csv('../data_raw/TMDB_Meta/movies_metadata.csv', low_memory=False)

movies = movies[movies['id'].str.isnumeric()]
movies['id'] = movies['id'].astype(int)

meta = movies[['id', 'title']].merge(credits, on='id')


def fetch_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return i['name']
    except: pass
    return ""

def fetch_top_cast(text):
    try:
        L = [i['name'] for i in ast.literal_eval(text)]
        return "|".join(L[:3])
    except: pass
    return ""

meta['director'] = meta['crew'].apply(fetch_director)
meta['cast'] = meta['cast'].apply(fetch_top_cast)
meta = meta[['title', 'director', 'cast']].drop_duplicates(subset=['title'])

master = pd.read_csv('../data_clean/clean_masterdata.csv')
master = master.merge(meta, on='title', how='left')
master.to_csv('../data_clean/clean_masterdata.csv', index=False)
