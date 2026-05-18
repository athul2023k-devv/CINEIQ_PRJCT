import pandas as pd
import ast

links = pd.read_csv('../data_raw/MvLns_25M DS/ml-25m/links.csv', low_memory=False)
tmdb_meta = pd.read_csv('../data_raw/TMDB_Meta/movies_metadata.csv', usecols=['id', 'title', 'genres', 'overview'],
                        low_memory=False)
# JUST TAKING THE ID, TITLE, GENRES AND OVERVIE FOR NOW
ratings = pd.read_csv('../data_raw/MvLns_25M DS/ml-25m/ratings.csv', low_memory=False)


def clean_genres(
        text):  # The function defined here will be handed in a raw text(from the genre column one row at a time)
    try:

        genres_listofdict = ast.literal_eval(
            text)  # CONVERTS THE UNEVALUABLE STRING (JSON) INTO PYTHON LIST OF DICTIONARIES
        genres_list = [genre_dict['name'] for genre_dict in genres_listofdict]
        return " ".join(genres_list)  # RETUNRNS ALL THE LIST VALUES WITHOUT THE COMMAS/BRACKETS
    except:
        return ("")


tmdb_meta['genres'] = tmdb_meta['genres'].apply(clean_genres)

tmdb_meta['id'] = pd.to_numeric(tmdb_meta['id'],errors = 'coerce') #WHAT THIS DOES IS THE COLUMN POINTS HAVING NON NUMBERS ARE FORCED TO BE BLANK(NaN)
tmdb_meta = tmdb_meta.dropna(subset = ['id'])
links = links.dropna(subset = ['tmdbId'])
merge_1 = pd.merge(tmdb_meta, links, left_on = 'id',right_on = 'tmdbId',how = 'inner' ) #Merges at the columns (left(id) of tmdb and right(tmdbId) of links)
final_merge = pd.merge(ratings,merge_1, on = 'movieId',how = 'inner')#How  = 'inner' dumps the unmatched values
#print(f"Dimenisons: {df.shape}")
#print(final_merge[['userId','title', 'rating']].head())
#print("The no.of blanks(NaN):",final_merge.isnull().sum())

#Filling the blanks with ''
final_merge = final_merge.dropna(subset =['title'])
final_merge['overview'] = final_merge['overview'].fillna('')
final_merge = final_merge.drop(columns = ['timestamp','id','tmdbId'])
#print("The no.of blanks(NaN):",final_merge.isnull().sum())

final_merge.to_csv('clean_masterdata.csv',index = False)