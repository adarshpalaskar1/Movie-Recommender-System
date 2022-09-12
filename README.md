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
