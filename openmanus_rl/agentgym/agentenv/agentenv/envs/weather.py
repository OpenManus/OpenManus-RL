from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class WeatherEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent. You can use actions to help people solve problems.\nWe detail name, description, input(parameters) and output(returns) of each action as follows:\nName: get_user_current_date()\nDescription: Get the user's current date.\nReturns:\nThe current date in 'YYYY-MM-DD' format.\n\nName: get_user_current_location()\nDescription: Get the user's current city.\nReturns:\nThe user's current city.\n\nName: get_historical_temp(latitude, longitude, start_date, end_date)\nDescription: Get historical temperature data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).\n- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).\nReturns:\nHistorical temperature data.\n\nName: get_historical_rain(latitude, longitude, start_date, end_date)\nDescription: Get historical rainfall data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).\n- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).\nReturns:\nHistorical rainfall data.\n\nName: get_historical_snow(latitude, longitude, start_date, end_date)\nDescription: Get historical snowfall data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).\n- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).\nReturns:\nHistorical snowfall data.\n\nName: get_snow_forecast(latitude, longitude, start_date, end_date)\nDescription: Get snowfall forecast data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).\n- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).\nReturns:\nSnowfall forecast data.\n\nName: get_current_snow(latitude, longitude, current_date)\nDescription: Get current snowfall data for a specified location and date.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- current_date (Type: string): The current date to retrieve snowfall data (YYYY-MM-DD).\nReturns:\nCurrent snowfall data.\n\nName: get_current_temp(latitude, longitude, current_date)\nDescription: Get current temperature data for a specified location and date.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- current_date (Type: string): The current date to retrieve temperature data (YYYY-MM-DD).\nReturns:\nCurrent temperature data.\n\nName: get_latitude_longitude(name)\nDescription: Get latitude and longitude information for a specified location name.\nParameters:\n- name (Type: string): The name of the location. (e.g., city name)\nReturns:\nlatitude and longitude information for the specified location.\n\nName: get_elevation(latitude, longitude)\nDescription: Get elevation data for a specified location.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\nReturns:\nElevation data for the specified location.\n\nName: get_temp_forecast(latitude, longitude, start_date, end_date)\nDescription: Get temperature forecast data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).\n- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).\nReturns:\nTemperature forecast data.\n\nName: get_rain_forecast(latitude, longitude, start_date, end_date)\nDescription: Get rainfall forecast data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the forecast (YYYY-MM-DD).\n- end_date (Type: string): The end date of the forecast (YYYY-MM-DD).\nReturns:\nRainfall forecast data.\n\nName: get_current_rain(latitude, longitude, current_date)\nDescription: Get current rainfall data for a specified location and date.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- current_date (Type: string): The current date to retrieve rainfall data (YYYY-MM-DD).\nReturns:\nCurrent rainfall data.\n\nName: get_distance(latitude1, longitude1, latitude2, longitude2)\nDescription: Calculate the distance between two sets of latitude and longitude coordinates.\nParameters:\n- latitude1 (Type: number): The latitude of the first location.\n- longitude1 (Type: number): The longitude of the first location.\n- latitude2 (Type: number): The latitude of the second location.\n- longitude2 (Type: number): The longitude of the second location.\nReturns:\nThe distance between the two sets of coordinates in kilometers.\n\nName: get_historical_air_quality_index(latitude, longitude, start_date, end_date)\nDescription: Get historical air quality index data for a specified location and date range.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- start_date (Type: string): The start date of the historical data (YYYY-MM-DD).\n- end_date (Type: string): The end date of the historical data (YYYY-MM-DD).\nReturns:\nHistorical air quality index (PM2.5) data.\n\nName: get_current_air_quality_index(latitude, longitude, current_date)\nDescription: Get current air quality index data for a specified location and date.\nParameters:\n- latitude (Type: number): The latitude of the location.\n- longitude (Type: number): The longitude of the location.\n- current_date (Type: string): The current date to retrieve air quality index data (YYYY-MM-DD).\nReturns:\nCurrent air quality index (PM2.5) data.\n\nName: get_air_quality_level(air_quality_index)\nDescription: Determine the air quality level based on the air quality index (AQI).\nParameters:\n- air_quality_index (Type: number): The air quality index (AQI) value.\nReturns:\nThe air quality level, which can be 'good', 'fair', 'moderate', 'poor', 'very poor', or 'extremely poor'.\n\nName: check_valid_actions()\nDescription: Get supported actions for current tool.\nReturns:\n- actions (Type: array): Supported actions for current tool.\n\nName: finish(answer)\nDescription: Return an answer and finish the task\nParameters:\n- answer (Type: ['string', 'number', 'array']): The answer to be returned\n\nIf you want to get the latitude and longitude information of a city, you must call \"get_latitude_longitude\"! Do not generate it by yourself which maybe wrong. If you are finished, you will call \"finish\" action.\nPlease refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!\n\nYour response must be in the format of:\nThought: [your thought]\n\nAction: [your action] with Action Input: [your action input]\n\nHere is an example:\n\nGoal: What is the lowest temperature yesterday?\nThought: This question is about the lowest temperature yesterday, I should first get the location information of the user.\n\nAction: get_user_current_location with Action Input: {}\nObservation: Shanghai\nThought: The user is currently in Shanghai. I should first get the latitude and longitude information of Shanghai.\n\nAction: get_latitude_longitude with Action Input: {\"name\": \"Shanghai\"}\nObservation: {'results': [{'name': 'Shanghai', 'latitude': 31.22222, 'longitude': 121.45806, 'country_code': 'CN'}, {'name': 'Shanghai', 'latitude': 34.85009, 'longitude': -87.08501, 'country_code': 'US'}, {'name': 'Cornelia', 'latitude': 38.64363, 'longitude': -93.73938, 'country_code': 'US'}]}\nThought: I have got the latitude and longitude information of Shanghai, I should get the current date to get the date of yesterday.\n\nAction: get_user_current_date with Action Input: {}\nObservation: 2015-01-02\nThought: Current date in 2015-01-02, so yesterday is 2015-01-01. Now, I can get the temperature data of Shanghai in 2015-01-01.\n\nAction: get_historical_temp with Action Input: {\"latitude\": 31.22222, \"longitude\": 121.45806, \"start_date\": \"2015-01-01\", \"end_date\": \"2015-01-01\"}\nObservation: {'latitude': 31.200005, 'longitude': 121.5, 'daily_units': {'time': 'iso8601', 'temperature_2m_max': '\u00b0C', 'temperature_2m_min': '\u00b0C', 'temperature_2m_mean': '\u00b0C'}, 'daily': {'time': ['2015-01-01'], 'temperature_2m_max': [4.3], 'temperature_2m_min': [-3.6], 'temperature_2m_mean': [-0.1]}}\nThought: The average temperature is -0.1, I will call finish to end the task.\n\nAction: finish with Action Input: {\"answer\": -0.1}\nObservation: -0.1\n",
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


class WeatherTask(BaseTask):
    env_client_cls = WeatherEnvClient
    env_name = "Weather"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
