import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CINEIQ AI", page_icon="🎬", layout="wide")


@st.cache_data
def load_user_data():
    df = pd.read_csv('../data_clean/clean_masterdata.csv')
    df = df.drop_duplicates(subset=['userId', 'movieId']).reset_index(drop=True)
    return df

df = load_user_data()

st.title("🎬 CINEIQ")
st.markdown("SENTIMENT AWARE AI RECOMMENDATIONS")
st.divider()
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recommendations_from_api(movie_title):
    try:
        response = requests.post("http://localhost:8000/recommend", json={"title": movie_title})
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return "NOT_FOUND"
        else:
            return f"API Error: {response.text}"
    except requests.exceptions.ConnectionError:
        return "CONNECTION_ERROR"

if "rec_results" not in st.session_state:
    st.session_state.rec_results = None
if "last_searched" not in st.session_state:
    st.session_state.last_searched = ""

tab1, tab2 = st.tabs(["AI Recommender", "User Taste Dashboard"])

with tab1:
    st.header("Find Your Next Movie")
    user_movie = st.text_input("Enter a movie you love:",
                               value=st.session_state.last_searched)

    if st.button("Generate Recommendations", type="primary"):
        if user_movie:
            st.session_state.last_searched = user_movie
            with st.spinner("Asking the CINEIQ API Engine..."):
                result = fetch_recommendations_from_api(user_movie)

                if result == "CONNECTION_ERROR":
                    st.error(" Could not connect to the API! ")
                elif isinstance(result, str) and result.startswith("API Error"):
                    st.error(result)
                else:
                    st.session_state.rec_results = result
        else:
            st.warning("Please type a movie title first!")


    if st.session_state.rec_results == "NOT_FOUND":
        st.error("Movie not found in the local catalog database. Try another title!")
    elif st.session_state.rec_results:
        st.success("Analysis Complete!")
        try:
            for rank, movie in enumerate(st.session_state.rec_results, 1):
                st.markdown(f"### {rank}. {movie['title']}")
                if "keywords:" in str(movie['lime_report']).lower():
                    st.info(f"**LIME  Readout:** {movie['lime_report']}")
                else:
                    st.info(f"**LIME Readout:** {movie['lime_report']}")
        except Exception as e:
            st.error(f"UI Drawing Error: {e}")
            st.write("Raw API Data received:", st.session_state.rec_results)


with tab2:
    st.header("User Taste Profiler")
    st.markdown("Analyze user rating histories to uncover unique cinematic DNA.")

    sample_users = df['userId'].unique()[:100]
    selected_user = st.selectbox("Select a User ID to analyze:", sample_users)

    if selected_user:
        user_history = df[df['userId'] == selected_user].dropna(subset=['genres'])
        user_history = user_history[user_history['genres'].astype(str).str.strip() != ""]

        st.subheader(f"Total Movies Rated with Genre Data: {len(user_history)}")

        if len(user_history) > 0:

            genres_expanded = user_history.assign(genres=user_history['genres'].astype(str).str.split('|')).explode(
                'genres')
            genre_counts = genres_expanded['genres'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']

            num_unique_genres = len(genre_counts)

            if num_unique_genres >= 3:
                st.markdown("### Genre DNA (Radar Profile)")
                top_genres = genre_counts.head(8)
                fig_radar = px.line_polar(top_genres, r='Count', theta='Genre', line_close=True, template="plotly_dark")
                fig_radar.update_traces(fill='toself', line_color='#00ff9d')
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.markdown("### Genre Preferences (Distribution Profile)")
                st.info(
                    " Note: A bar layout was selected adaptively because this user profile has highly concentrated genre choices.")
                fig_bar = px.bar(genre_counts, x='Count', y='Genre', orientation='h', template="plotly_dark",
                                 color_discrete_sequence=['#00ff9d'])
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()


            st.markdown("### Cinematic Era Preferences")


            if 'year' in user_history.columns:
                decade_data = user_history.dropna(subset=['year']).copy()

                if len(decade_data) > 0:

                    decade_data['decade'] = (decade_data['year'] // 10 * 10).astype(int).astype(str) + "s"
                    decade_counts = decade_data['decade'].value_counts().reset_index()
                    decade_counts.columns = ['Decade', 'Movies Watched']
                    decade_counts = decade_counts.sort_values('Decade')

                    fig_decade = px.bar(
                        decade_counts,
                        x='Decade',
                        y='Movies Watched',
                        template="plotly_dark",
                        color_discrete_sequence=['#ff0055']
                    )
                    fig_decade.update_layout(xaxis_title="Decade", yaxis_title="Number of Movies")
                    st.plotly_chart(fig_decade, use_container_width=True)
                else:
                    st.info(" Not enough year data to generate a decade profile for this user.")
            else:
                st.warning(" Year data missing from CSV! Did you run the add_year.py script?")

            st.divider()


            st.markdown("### Hollywood Affinities")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top Directors**")
                if 'director' in user_history.columns:
                    directors = user_history.dropna(subset=['director'])
                    if len(directors) > 0:
                        dir_counts = directors['director'].value_counts().head(5).reset_index()
                        dir_counts.columns = ['Director', 'Count']
                        fig_dir = px.bar(dir_counts, x='Count', y='Director', orientation='h', template="plotly_dark",
                                         color_discrete_sequence=['#ffb703'])
                        fig_dir.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_dir, use_container_width=True)
                    else:
                        st.info("Not enough data to calculate Director affinities.")
                else:
                    st.warning("Director data missing from CSV.")

            with col2:
                st.markdown("**Favorite Cast Members**")
                if 'cast' in user_history.columns:
                    cast_data = user_history.dropna(subset=['cast'])
                    if len(cast_data) > 0:
                        cast_expanded = cast_data.assign(cast=cast_data['cast'].astype(str).str.split('|')).explode(
                            'cast')
                        cast_counts = cast_expanded['cast'].value_counts().head(5).reset_index()
                        cast_counts.columns = ['Actor', 'Count']
                        fig_cast = px.bar(cast_counts, x='Count', y='Actor', orientation='h', template="plotly_dark",
                                          color_discrete_sequence=['#8ecae6'])
                        fig_cast.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_cast, use_container_width=True)
                    else:
                        st.info("Not enough data to calculate Actor affinities.")
                else:
                    st.warning("Cast data missing from CSV.")

        else:
            st.warning(
                " This specific user profile has no associated genre metadata records available. Try selecting another User ID!")