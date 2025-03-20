import requests
import copy
from copy import deepcopy
from typing import List, Dict, Any, Union
import os
from dotenv import load_dotenv

load_dotenv()

url = 'https://api.themoviedb.org/3/search/movie'

URLS = {
    "movies" : {
        "details" : "https://api.themoviedb.org/3/movie/{movie_id}",
        "credits" : "https://api.themoviedb.org/3/movie/{movie_id}/credits",
        "keywords" : "https://api.themoviedb.org/3/movie/{movie_id}/keywords",
        "alternative_titles": "https://api.themoviedb.org/3/movie/{movie_id}/alternative_titles",
        "translation": "https://api.themoviedb.org/3/movie/{movie_id}/translations"
    },
    "people" : {
        "details" : "https://api.themoviedb.org/3/person/{person_id}",
        "movie_credits" : "https://api.themoviedb.org/3/person/{person_id}/movie_credits",
        "external_ids": "https://api.themoviedb.org/3/person/{person_id}/external_ids",
    },
    "search" : {
        "movie" : "https://api.themoviedb.org/3/search/movie",
        "person": "https://api.themoviedb.org/3/search/person"
    }
}
params = {
    'query': 'Mulholland Drive',
    'include_adult': 'false',
    'language': 'en-US',
    'page': '1'
}

headers = {
    'Authorization': 'Bearer {}'.format(os.environ["MOVIE_KEY"] if "MOVIE_KEY" in os.environ.keys() else ""),
}

def clean_observation(
                        observation : Union[ List[Dict[str, Any]], Dict[str, Any] ]
                    ):
    # remove all "id" in observation
    new_observation = deepcopy(observation)
    ignore_keys = ["overview", "biography", "vote_average", "genres", "revenue", "budget", "release_date"]
    if isinstance(new_observation, list):
        for item in new_observation:
            if isinstance(item, dict):
                for key in ignore_keys:
                    if key in item.keys():
                        item.pop(key)
    elif isinstance(new_observation, dict):
        for key in ignore_keys:
            if key in new_observation:
                new_observation.pop(key)
    return new_observation

def log_path(func):
    def wrapper(*args, **kwargs):
        if "action_path" in kwargs.keys():
            action_path = kwargs["action_path"]
            kwargs.pop("action_path")
            success, result = func(*args, **kwargs)

            # convert value in kwargs to string
            # for key, value in kwargs.items():
                # kwargs[key] = str(value)

            if success:
                action_path.append({
                    "Action" : func.__name__,
                    "Action Input" : str(kwargs),
                    "Observation": result,
                    "Subgoal": clean_observation(result) 
                })
                return result
            else:
                action_path.append({
                    "Action" : func.__name__,
                    "Action Input" : str(kwargs),
                    "Observation": result,
                    "Subgoal": "Calling " + func.__name__ + " with " + str(kwargs) + " failed",
                })
                return result
        else:
            return func(*args, **kwargs)
    return wrapper

class movie_toolkits:
    def __init__(self):
        pass

    @log_path
    def get_search_movie(self, movie_name=None):
        url = URLS['search']['movie']
        params['query'] = movie_name
        params['include_adult'] = 'false'
        params['language'] = 'en-US'
        params['page'] = '1'
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("search_movie.json", "w"), indent=4)
            return_data = [{
                "id" : data["results"][0]["id"],
                "overview" : data["results"][0]["overview"],
                "title": data["results"][0]["title"],
                # "release_date": data["results"][0]["release_date"]
            }]
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_movie_details(self, movie_id=None):
        url = URLS['movies']['details'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_details.json", "w"), indent=4)
            return_data = {
                "title": data["title"],
                "budget": data["budget"],
                "genres": data["genres"],
                "revenue": data["revenue"],
                "vote_average": data["vote_average"],
                "release_date": data["release_date"],
                # "production_companies": data["production_companies"],
                # "production_countries": data["production_countries"],
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_movie_production_companies(self, movie_id=None):
        url = URLS['movies']['details'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_details.json", "w"), indent=4)
            return_data = {
                "production_companies": data["production_companies"],
            }
            return True, return_data
        else:
            return False, response.text
    
    @log_path
    def get_movie_production_countries(self, movie_id=None):
        url = URLS['movies']['details'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_details.json", "w"), indent=4)
            return_data = {
                "production_countries": data["production_countries"],
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_movie_cast(self, movie_id=None):
        url = URLS['movies']['credits'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            
            # json.dump(data, open("movie_credits.json", "w"), indent=4)
            return_data = {
                "cast" : [ {"id": cast["id"], "name": cast["name"], "character": cast["character"]} for cast in data["cast"][:10] ],
            }
            return True, return_data
        else:
            return False, response.text
        
    @log_path
    def get_movie_crew(self, movie_id=None):
        url = URLS['movies']['credits'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            
            must_contain_job = ["Director", "Producer", "Writer"]
            
            # json.dump(data, open("movie_credits.json", "w"), indent=4)
            return_data = {
                "crew": [ {"id": crew["id"], "name": crew["name"], "job": crew["job"]} for crew in data["crew"] if crew["job"] in must_contain_job ]
            }
            if len(return_data["crew"]) < 10:
                ids = [ crew["id"] for crew in return_data["crew"] ]
                for crew in data["crew"]:
                    if crew["id"] not in ids:
                        return_data["crew"].append({"id": crew["id"], "name": crew["name"], "job": crew["job"]})
                    if len(return_data["crew"]) == 10:
                        break
            # print("Out of Loop")
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_movie_keywords(self, movie_id=None):
        url = URLS['movies']['keywords'].format(movie_id=movie_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_keywords.json", "w"), indent=4)
            return_data = {
                "keywords" : [ keyword["name"] for keyword in data["keywords"] ]
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_search_person(self, person_name=None):
        url = URLS['search']['person']
        params['query'] = person_name
        params['include_adult'] = 'false'
        params['language'] = 'en-US'
        params['page'] = '1'
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # print(data)
            # json.dump(data, open("search_people.json", "w"), indent=4)
            if len(data["results"]) == 0:
                return True, []
            else:
                return_data = [{
                    "id" : data["results"][0]["id"],
                    "name": data["results"][0]["name"],
                    # "gender": data["results"][0]["gender"]
                }]
                return True, return_data
        else:
            return False, response.text

    @log_path
    def get_person_details(self, person_id=None):
        url = URLS['people']['details'].format(person_id=person_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("people_details.json", "w"), indent=4)
            #! no imdb information
            return_data = {
                "name": data["name"],
                "biography": data["biography"],
                "birthday": data["birthday"],
                "place_of_birth": data["place_of_birth"],
            }
            return True, return_data
        else:
            return False, response.text
        
    @log_path
    def get_person_cast(self, person_id=None):
        url = URLS['people']['movie_credits'].format(person_id=person_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("people_movie_credits.json", "w"), indent=4)
            return_data = {
                "cast" : [ {"id": cast["id"], "title": cast["title"], "character": cast["character"]} for cast in data["cast"][:10] ],
            }
            return True, return_data
        else:
            return False, response.text
        
    @log_path
    def get_person_crew(self, person_id=None):
        url = URLS['people']['movie_credits'].format(person_id=person_id)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("people_movie_credits.json", "w"), indent=4)
            return_data = {
                "crew": [ {"id": crew["id"], "title": crew["title"], "job": crew["job"]} for crew in data["crew"][:10] ]
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_person_external_ids(self, person_id=None):
        url = URLS['people']['external_ids'].format(
            person_id=person_id
        )
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("people_external_ids.json", "w"), indent=4)            
            return_data = {
                "imdb_id": data["imdb_id"],
                "facebook_id": data["facebook_id"],
                "instagram_id": data["instagram_id"],
                "twitter_id": data["twitter_id"]
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_movie_alternative_titles(self, movie_id=None):
        url = URLS['movies']['alternative_titles'].format(
            movie_id=movie_id
        )
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_alternative_titles.json", "w"), indent=4, ensure_ascii=False)            
            return_data = data
            return True, return_data
        else:
            return False, response.text
    
    @log_path
    def get_movie_translation(self, movie_id=None):
        url = URLS["movies"]["translation"].format(
            movie_id=movie_id
        )
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # json.dump(data, open("movie_translation.json", "w"), indent=4, ensure_ascii=False)            
            
            # get dutch
            target_lang = ["NL", "CN", "US", "DE", "RU", "JP"]
            return_data = copy.deepcopy(data)

            return_data["translations"] = [ item for item in data["translations"] if item["iso_3166_1"] in target_lang ]

            for item in return_data["translations"]:
                item["data"].pop("title")

            return True, return_data
        else:
            return False, response.text

    @log_path
    def finish(self, answer):
        if type(answer) == list:
            answer = sorted(answer)
        return True, answer


if __name__ == "__main__":
    tool = movie_toolkits()
    print( tool.get_movie_details("934433") )