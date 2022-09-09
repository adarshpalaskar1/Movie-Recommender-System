import requests

def get_poster_and_safe_mode(tmdbId):

    response_ = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=70fab75d084560da6c7ac9cedef4605b'.format(tmdbId))
    data = response_.json()

    is_adult_present = 'adult' in data
    is_poster_present = 'poster_path' in data

    is_safe = False

    if(is_poster_present == False):
        return False, False

    if(is_adult_present and data['adult'] == False):
        is_safe = True

    return 'https://image.tmdb.org/t/p/w500/' + data['poster_path'], is_safe

