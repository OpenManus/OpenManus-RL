from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class MovieEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent. You can use actions to help people solve problems.\nWe detail name, description, input(parameters) and output(returns) of each action as follows:\nName: get_search_movie(movie_name)\nDescription: Search for a movie by name and return basic details\nParameters:\n- movie_name (Type: string): The name of the movie to search for.\nReturns:\n- id : The ID of the found movie.\n- overview : The overview description of the movie.\n- title : The title of the movie.\n\nName: get_movie_details(movie_id)\nDescription: Get detailed information about a movie by ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- budget : The budget of the movie.\n- genres : The genres of the movie.\n- revenue : The revenue of the movie.\n- vote_average : The average vote score of the movie.\n- release_date : The release date of the movie.\n\nName: get_movie_production_companies(movie_id)\nDescription: Get the production companies of a movie by its ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- production_companies : The production companies of the movie.\n\nName: get_movie_production_countries(movie_id)\nDescription: Get the production countries of a movie by its ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- production_countries : The production countries of the movie.\n\nName: get_movie_cast(movie_id)\nDescription: Retrieve the list of the top 10 cast members from a movie by its ID.\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- cast : List of the top 10 cast members.\n\nName: get_movie_crew(movie_id)\nDescription: Retrieve the list of crew members (limited to 10) from a movie by its ID. The list primarily includes Director, Producer, and Writer roles.\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- crew : List of the top 10 of crew members\n\nName: get_movie_keywords(movie_id)\nDescription: Get the keywords associated with a movie by ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- keywords : The keywords associated with the movie.\n\nName: get_search_person(person_name)\nDescription: Search for a person by name.\nParameters:\n- person_name (Type: string): The name of the person to search for.\nReturns:\n- id : The ID of the found person.\n- name : The name of the person.\n\nName: get_person_details(person_id)\nDescription: Get detailed information about a person by ID\nParameters:\n- person_id (Type: string): The ID of the person.\nReturns:\n- biography : The biography of the person.\n- birthday : The birthday of the person.\n- place_of_birth : The place of birth of the person.\n\nName: get_person_cast(person_id)\nDescription: Retrieve the top 10 movie cast roles of a person by their ID\nParameters:\n- person_id (Type: string): The ID of the person.\nReturns:\n- cast : A list of movies where the person has acted, limited to top 10\n\nName: get_person_crew(person_id)\nDescription: Retrieve the top 10 movie crew roles of a person by their ID\nParameters:\n- person_id (Type: string): The ID of the person.\nReturns:\n- crew : A list of movies where the person has participated as crew, limited to top 10\n\nName: get_person_external_ids(person_id)\nDescription: Get the external ids for a person by ID\nParameters:\n- person_id (Type: string): The ID of the person.\nReturns:\n- imdb_id : The IMDB id of the person.\n- facebook_id : The Facebook id of the person.\n- instagram_id : The Instagram id of the person.\n- twitter_id : The Twitter id of the person.\n\nName: get_movie_alternative_titles(movie_id)\nDescription: Get the alternative titles for a movie by ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- titles : The alternative titles of the movie.\n- id : The ID of the movie.\n\nName: get_movie_translation(movie_id)\nDescription: Get the description translation for a movie by ID\nParameters:\n- movie_id (Type: string): The ID of the movie.\nReturns:\n- translations : The description translation of the movie.\n- id : The ID of the movie.\n\nName: check_valid_actions()\nDescription: Get supported actions for current tool.\nReturns:\n- actions (Type: array): Supported actions for current tool.\n\nName: finish(answer)\nDescription: Return an answer and finish the task\nParameters:\n- answer (Type: ['string', 'number', 'array']): The answer to be returned\n\nIf you want to get the movie_id or person_id, Please call \"get_search_person\", \"get_search_movie\"! Do not generate it by yourself which maybe wrong. If you are finished, you will call \"finish\" action. \nPlease refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!\n\nYour response must be in the format of:\nThought: [your thought]\n\nAction: [your action] with Action Input: [your action input]\n\nHere is an example:\n\nGoal: When did the movie Scream 6 come out?\nThought: I need to know the ID of the movie Scream 6 first.\n\nAction: get_search_movie with Action Input: {\"movie_name\": \"Scream 6\"}\nObservation: {'id': 934433, 'overview': 'Following the latest Ghostface killings, the four survivors leave Woodsboro behind and start a fresh chapter.', 'title': 'Scream VI'}\nThought: I can get the release date from get_movie_details know.\n\nAction: get_movie_details with Action Input: {\"movie_id\": \"934433\"}\nObservation: {'budget': 35000000, 'genres': [{'id': 27, 'name': 'Horror'}, {'id': 53, 'name': 'Thriller'}, {'id': 9648, 'name': 'Mystery'}], 'revenue': 168961389, 'vote_average': 7.175, 'release_date': '2023-03-08'}\nThought: The release date is 2023-03-08.\n\nAction: finish with Action Input: {\"answer\": \"2023-03-08\"}\nObservation: 2023-03-08\n",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response


class MovieTask(BaseTask):
    env_client_cls = MovieEnvClient
    env_name = "Movie"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
