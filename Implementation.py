#!/usr/bin/env python
# coding: utf-8

# ## **Project Overview**

# In[ ]:


from IPython.display import Image
Image(filename='/content/Project_Overview.png') 


# ## **User-User Collaborative Filtering using Nearest Neighbours**

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Importing all the necessary data files.
links = pd.read_csv('/content/links.csv')
movies = pd.read_csv('/content/movies.csv')
ratings = pd.read_csv('/content/ratings.csv')
tags = pd.read_csv('/content/tags.csv')

display(links)
print()

display(movies)
print()

display(ratings)
print()

display(tags)
print()


# In[79]:


# From the ratings dataset, we will create another dataset, where, for each movie, 
# all the ratings given by the 610 users are displayed.
df = ratings.pivot(index='movieId',columns='userId',values='rating')
display(df)
print()

# The NaN values correspond to the users that have not rated a particular movie.
# We will replace them with zeroes to create a sparse matrix.
df = df.fillna(0)
display(df)


# In[80]:


# This will give us the number of users who have voted for a particular movie
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')

sns.set_palette("muted")
f,ax = plt.subplots(1,1,figsize=(12,5))
sns.scatterplot(no_user_voted.index,no_user_voted)
plt.xlabel('MovieId')
plt.ylabel('No. of users')
plt.title("No. of users that have rated a particular movie")
plt.show()


# In[81]:


# This will give us the number of movies a user has rated
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

f,ax = plt.subplots(1,1,figsize=(12,5))
sns.scatterplot(no_movies_voted.index,no_movies_voted)
plt.xlabel('UserId')
plt.ylabel('No. of movies rated')
plt.title("No. of movies that a particular user has rated")
plt.show()


# In[82]:


display(df)

# Plotting the distribution of ratings
fig, ax = plt.subplots(figsize=(12,5))
ax.set_title('Distribution of ratings')
sns.countplot(ratings['rating'])
ax.set_xlabel("Rating")
ax.set_ylabel("Total number of ratings")
plt.show()


# In[83]:


# We have obtained a dataset of shape (2121,378) after the previous steps. However, 
# most of the values are still zero. We will attempt to reduce the sparsity in the dataset.
from scipy.sparse import csr_matrix

data = csr_matrix(df.values)
df.reset_index(inplace=True)

display(df)


# In[84]:


# To make the movie recommendation model, we will use Nearest Neighbours.
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=10, algorithm='auto', n_jobs=-1)
knn.fit(data)

# We will first take a particular movie input movie from the user
# We will then output 10 movies with the most similarities to the input movie.
def recommend(inp):
  movie_list = movies[movies['title'].str.contains(inp)]
  if(len(movie_list)==0):
    print("This movie is not in the database. Try another one!")
    return
  movie_idx= movie_list.iloc[0]['movieId']
  movie_idx = df[df['movieId'] == movie_idx].index[0]
  distances , indices = knn.kneighbors(data[movie_idx],n_neighbors=11)    
  rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
  recommend_frame = []
  for val in rec_movie_indices:
    movie_idx = df.iloc[val[0]]['movieId']
    idx = movies[movies['movieId'] == movie_idx].index
    recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]/100})
  recommendations = pd.DataFrame(recommend_frame,index=range(1,11))
  return recommendations


# In[85]:


# Testing the recommendation system
input = "Iron Man"
print("The top ten movies similar to", input, "are -")
recommend(input)


# ## **User-User and Item-Item based Collaborative Filtering using Pearson and Cosine Similarity**

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


ratings = pd.read_csv('/content/ratings.csv')


# In[ ]:


ratings


# In[ ]:


ratings.drop('timestamp', axis = 1, inplace = True)


# In[ ]:


ratings


# In[ ]:


# From the ratings dataset, we will create another dataset, where, for each movie, 
# all the ratings given by the 610 users are displayed.
ratings = ratings.pivot(index='movieId',columns='userId',values='rating')

# The NaN values correspond to the users that have not rated a particular movie.
# We will replace them with zeroes to create a sparse matrix.
ratings = ratings.fillna(0)
display(ratings)


# In[ ]:


# Shuffling the data and dividing into train and test sets, where size of
# test set = 0.2
ratings = ratings.sample(frac = 1) 

train_data = ratings[0:7780]
test_data = ratings[7780:]


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data.values
test_data_matrix = test_data.values

# Check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)


# Using Pearson similarity and calculating pairwise distances:

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[ ]:


user_correlation.shape


# In[ ]:


# Visualization of similarity in user behaviours.
# Darker colour represents that the users are more similar.

import seaborn as sns

user_correlation_reduced = user_correlation[0:50, 0:50]

sns.set(rc={'figure.figsize':(10,10)})

sns.heatmap(user_correlation_reduced, cmap="YlGnBu")


# In[ ]:


# Item Similarity Matrix
from sklearn.metrics.pairwise import pairwise_distances

item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[ ]:


item_correlation.shape


# In[ ]:


# Visualization of similarity in movie genres.
# Darker colour represents that the movies are more similar.

item_correaltion_reduced = item_correlation[0:50, 0:50]

sns.set(rc={'figure.figsize':(10,10)})

sns.heatmap(item_correaltion_reduced, cmap="YlGnBu")


# In[ ]:


# Function to predict ratings
def predict(ratings, similarity, type='user'):

    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) #np.newaxis is used to add and extra column to store predictions

        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    
    return pred


# Evaluating the model:

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt


def rmse(pred, actual):
    
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    
    return sqrt(mean_squared_error(pred, actual))


# In[ ]:



user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

print('RMSE for Train data :')

print('User-based CF RMSE: ', rmse(user_prediction, test_data_matrix))
print('Item-based CF RMSE: ', rmse(item_prediction, test_data_matrix))


# In[ ]:


print('RMSE for Test data :')

print('User-based CF RMSE: ' ,rmse(user_prediction, train_data_matrix))
print('Item-based CF RMSE: ' ,rmse(item_prediction, train_data_matrix))


# Using Cosine Similarity:

# In[ ]:



# User Similarity Matrix
user_correlation_2 = 1 - pairwise_distances(train_data, metric='cosine')
user_correlation_2[np.isnan(user_correlation_2)] = 0
print(user_correlation_2)


# In[ ]:


# Item Similarity Matrix

item_correlation_2 = 1 - pairwise_distances(train_data_matrix.T, metric='cosine')
item_correlation_2[np.isnan(item_correlation_2)] = 0
print(item_correlation_2)


# Evaluating the model:

# In[ ]:


user_prediction_2 = predict(train_data_matrix, user_correlation_2, type='user')
item_prediction_2 = predict(train_data_matrix, item_correlation_2, type='item')

print('RMSE for Train data :')

print('User-based CF RMSE: ', rmse(user_prediction_2, test_data_matrix))
print('Item-based CF RMSE: ', rmse(item_prediction_2, test_data_matrix))


# In[ ]:


print('RMSE for Test data :')

print('User-based CF RMSE: ' ,rmse(user_prediction_2, train_data_matrix))
print('Item-based CF RMSE: ' ,rmse(item_prediction_2, train_data_matrix))


# In[ ]:





# ## **Content Based model using Term Frequency (TF), Inverse Document Frequency (IDF) and Cosine Similarity**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib.pyplot as plt


# In[ ]:


movies = pd.read_csv('/content/movies.csv')


# In[ ]:


movies


# Visualizaing the frequency of different words in movie titles:

# In[ ]:


# Import new libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Create a wordcloud of the movie titles
movies['title'] = movies['title'].fillna("").astype('str')
title_corpus = ' '.join(movies['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


# Cleaning the genre column and converting the values to string

movies['genres'] = movies['genres'].str.split('|')

movies['genres'] = movies['genres'].fillna("").astype('str')

movies


# Visualizaing the frequency of different Genres present in the dataset:

# In[ ]:



# Create a wordcloud of the movie genres

title_corpus = ' '.join(movies['genres'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[ ]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape


# In[ ]:



cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim


# In[ ]:


# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def recommendations_based_on_genre(title):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]


# In[ ]:


# Plot to represent all the vectors in 2 dimensions using PCA

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])       

X = pipeline.fit_transform(movies['genres']).todense()

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
sns.scatterplot(data2D[:,0], data2D[:,1] )


# In[ ]:


recommendations_based_on_genre('Good Will Hunting (1997)').head(10)


# In[ ]:


recommendations_based_on_genre('Jumanji (1995)').head(10)


# In[ ]:


recommendations_based_on_genre('Iron Man (2008)').head(10)


# In[ ]:





# # Hybrid Filtering using Linear Regression

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# In[ ]:


#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies['year'] = movies.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies['title'] = movies['title'].apply(lambda x: x.strip())
movies.head()


# In[ ]:


#Every genre is separated by a | so we simply have to call the split function on |
movies['genres'] = movies.genres.str.split('|')
movies.head()


# In[ ]:


#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()


# In[ ]:


#Drop removes a specified row or column from a dataframe
ratings = ratings.drop('timestamp', 1)
ratings.head()


# In[ ]:


userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies


# In[ ]:


#Filtering out the movies by title
inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies


# In[ ]:


#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies


# In[ ]:


#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable


# In[ ]:


inputMovies['rating']


# In[ ]:


#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
userProfile


# In[ ]:


#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()


# In[ ]:


genreTable.shape


# In[ ]:


#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()


# In[ ]:


#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()


# In[ ]:


#The final recommendation table
movies.loc[movies['movieId'].isin(recommendationTable_df.head(10).keys())]

