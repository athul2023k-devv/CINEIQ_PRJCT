import pandas as pd


movies = pd.read_csv('../data_raw/TMDB_Meta/movies_metadata.csv', low_memory=False)
master = pd.read_csv('../data_clean/clean_masterdata.csv')

movies['year'] = movies['release_date'].astype(str).str.slice(0, 4)
movies['year'] = pd.to_numeric(movies['year'], errors='coerce')


years_df = movies[['title', 'year']].dropna().drop_duplicates(subset=['title'])


if 'year' in master.columns:
    master = master.drop(columns=['year'])


master = master.merge(years_df, on='title', how='left')

master.to_csv('../data_clean/clean_masterdata.csv', index=False)
