# Movie Recommender using Multiple user inputs and ratings and TMDB API

Recommendation system built using multiple ML models that aims to 
predict users' interests based on their past behavior and 
preferences.

It uses MovieLens data containing two hundred thousand reviews 
on about 10000 different movies. The n-dimensional linear 
regression ML model or a single-layer neural network then 
predicts what movies the user may like using the user profile 
and the data of movie ratings. This web app has been deployed 
using the Streamlit framework. The code uses TMDB API, which 
fetches posters for each movie recommendation using its TMDB id. 
The project also has a child-safe mode, that filters out adult 
movies using the API data and also filters out crime and adult 
genre movies using the dataset. The code also contains three 
more ML models that use K-nearest neighbors, TF-IDF 
vectorization, and Cosine and Pearson similarities.

The application is deployed at - https://movie-recommender-userprofile.herokuapp.com/

Streamlit - https://streamlit.io/

TMDB API - https://developers.themoviedb.org/3/getting-started/introduction


## How to get the API key?

Create an account in https://www.themoviedb.org/, click on the 
API link from the left hand sidebar in your account settings and 
fill all the details to apply for API key. If you are asked for 
the website URL, just give "NA" if you don't have one. You will 
see the API key in your API sidebar once your request is approved.

## How to run this project on your local system?

1. Clone this repository in your local system.

2. run pip install -r requirements.txt in the terminal to install
the dependencies.(Your need to have python installed in your system).

3. Open the terminal and navigate to the project directory and run
streamlit run app.py and the app will run on the local host displayed
in the terminal.

4. Copy paste this link into the address bar of a web-browser.

5. Yay! You can now test the code and raise an issue if you find any
bugs or suggestions.

## Application Architecture
![Architecture](https://user-images.githubusercontent.com/83298237/190175048-f387d662-9385-49fd-9a6b-dc95bacf1a0b.png)

## Overview of ML Algorithms used
![Project_Overview](https://user-images.githubusercontent.com/83298237/190177646-422a177d-749b-4294-b7d1-28d1d5d51918.png)

1. Collaborative Filtering - Collaborative filtering approaches build a model from the user’s past
behaviour as well as similar decisions made by other users. This model is then used to predict
items that users may have an interest in.

2. Content based Filtering - Content-based filtering approaches use discrete characteristics of
an item in order to recommend additional items with similar properties. They are totally
based on a description of the item and a profile of the user’s preferences. It recommends
items based on the user’s past preferences.

For complete implementation details and outputs refer 
to this [file.](https://github.com/adarshpalaskar1/Movie-Recommender-System/blob/main/Implementation%20-%20Jupyter%20Notebook.pdf)


## Demo

You can choose to select the movies according to your preferred answer for all the questions, or simply put any movie you like or hate which will be
indicated to the ML model by the rating you enter. 

![Movie-Recommendation-System](https://user-images.githubusercontent.com/83298237/190182554-3719a198-e1e5-455d-9139-a30e564d12fb.png)



![Movie-Recommendation-System](https://user-images.githubusercontent.com/83298237/190182554-3719a198-e1e5-455d-9139-a30e564d12fb.png)
