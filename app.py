from copyreg import pickle
from urllib import request
import streamlit as st
import pickle
import pandas as pd
import numpy as np

from api_ import get_poster_and_safe_mode
from binary_search import binary_search_

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",

    menu_items={
        'Report a bug': "https://github.com/adarshpalaskar1/Movie-Recommendation-System/issues",
        'About': "This webapp uses ML algorithms to recommend movies based on your user profile and 2 hundred thousand ratings on 10000 different movies. It uses the TMDB api and the Movie Lens Dataset."
    }
)

st.title('Movie Recommender System')

st.subheader('Enter 5 different movies and rate them to create your Profile:')

movie_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)

ratings_dict = pickle.load(open('ratings.pkl', 'rb'))
ratings = pd.DataFrame(ratings_dict)

links_dict = pickle.load(open('links.pkl', 'rb'))
links = pd.DataFrame(links_dict)

#movie 1
option1 = st.selectbox(
    '1. What is your favourite Movie?',
    movies['title'].values)

st.write('You selected:', option1)

rating1 = st.slider('1. How would you rate this movie', 0, 5, 3)

#movie 2
option2 = st.selectbox(
     '2. What movie do you hate the most?',
     (movies['title'].values))

st.write('You selected:', option2)

rating2 = st.slider('2. How would you rate this movie', 0, 5, 3)

#movie 3
option3 = st.selectbox(
     '3. Which movie has most of the Genres you like?',
     (movies['title'].values))

st.write('You selected:', option3)

rating3 = st.slider('3. How would you rate this movie', 0, 5, 2)

#movie 4
option4 = st.selectbox(
     '4. Which movie has most of the Genres you hate?',
     (movies['title'].values))

st.write('You selected:', option4)

rating4 = st.slider('4. How would you rate this movie', 0, 5, 3)

#movie 5
option5 = st.selectbox(
     '5. What movie would you recommend to your friends?',
     (movies['title'].values))

st.write('You selected:', option5)

rating5 = st.slider('5. How would you rate this movie', 0, 5, 4)


# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)

child_safe_mode = st.checkbox('Enable child safe mode')

if st.button('Recommend movies for me'):
    

    userInput = [
            {'title': option1, 'rating': rating1},
            {'title': option2, 'rating': rating2},
            {'title': option3, 'rating': rating3},
            {'title': option4, 'rating': rating4},
            {'title': option5, 'rating': rating5}
         ] 
    inputMovies = pd.DataFrame(userInput)

    length_set = len({option1, option2, option3, option4, option5})

    if(length_set < 5):
        st.subheader("Please choose 5 different movies")
    
    else:

        st.header('Movies you may like: ')

        # Filtering out the movies by title
        inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]
        # Then merging it so we can get the movieId. It's implicitly merging it by title.
        inputMovies = pd.merge(inputId, inputMovies)
        # Dropping information we won't use from the input dataframe
        inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

        # Filtering out the movies from the input
        userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

        # Resetting the index to avoid future issues
        userMovies = userMovies.reset_index(drop=True)
        # Dropping unnecessary issues due to save memory and to avoid issues
        userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

        # Dot produt to get weights
        userProfile = (userGenreTable.transpose()).dot(inputMovies['rating'])

        # Removing genres to implement Child Safe Mode
        if(child_safe_mode):

            userProfile['Crime'] = 0
            userProfile['Romance'] = 0
            userProfile['War'] = 0



        # Now let's get the genres of every movie in our original dataframe
        genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
        # And drop the unnecessary information
        genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

        recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

        recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

        # The final recommendation table
        final_table = movies.loc[movies['movieId'].isin(recommendationTable_df.head(50).keys())]

        tmdbid = 0

        recommended_movies = []
        posters = []
        child_safe = []

        for movieid in final_table['movieId']:

            idx = binary_search_(links['movieId'], movieid, 0, 9742)   

            tmdbid = links.loc[idx][1]

            poster , is_safe = get_poster_and_safe_mode(int(tmdbid))
            
            if(poster == False):
                continue

            if(is_safe == False and child_safe_mode == True):
                continue
            
            child_safe.append(is_safe)
            posters.append(poster)
            
            recommended_movies.append(final_table.loc[idx][1])

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(recommended_movies[0])
            st.image(posters[0])

        with col2:
            st.text(recommended_movies[1])
            st.image(posters[1])

        with col3:
            st.text(recommended_movies[2])
            st.image(posters[2])

        with col4:
            st.text(recommended_movies[3])
            st.image(posters[3])

        with col5:
            st.text(recommended_movies[4])
            st.image(posters[4])


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(recommended_movies[5])
            st.image(posters[5])

        with col2:
            st.text(recommended_movies[6])
            st.image(posters[6])

        with col3:
            st.text(recommended_movies[7])
            st.image(posters[7])

        with col4:
            st.text(recommended_movies[8])
            st.image(posters[8])

        with col5:
            st.text(recommended_movies[9])
            st.image(posters[9])

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(recommended_movies[10])
            st.image(posters[10])

        with col2:
            st.text(recommended_movies[11])
            st.image(posters[11])

        with col3:
            st.text(recommended_movies[12])
            st.image(posters[12])

        with col4:
            st.text(recommended_movies[13])
            st.image(posters[13])

        with col5:
            st.text(recommended_movies[14])
            st.image(posters[14])


        



        



