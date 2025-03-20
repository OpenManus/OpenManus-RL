import requests
import logging
from geopy.distance import geodesic
from datetime import datetime
from copy import deepcopy
from datetime import datetime, timedelta

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# 从IP中生成物理地址

URLS = {
    "historical_weather": "https://archive-api.open-meteo.com/v1/archive",
    "geocoding": "https://geocoding-api.open-meteo.com/v1/search",
    "air_quality": "https://air-quality-api.open-meteo.com/v1/air-quality",
    "elevation": "https://api.open-meteo.com/v1/elevation",
    # "zipcode": "http://ZiptasticAPI.com/{zipcode}"
    # "weather_forecast": "https://api.open-meteo.com/v1/forecast",
}

def is_within_30_days(start_date: str, end_date: str) -> bool:
    # 将字符串格式的日期转换为datetime对象
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 计算两个日期之间的差异
    difference = end_date_obj - start_date_obj
    
    # 判断差异是否为30天
    if difference > timedelta(days=30):
        return False
    else:
        return True

def clean_observation(observation):
    new_observation = deepcopy(observation)
    if type(new_observation) == dict and "daily" in new_observation.keys() and "temperature_2m_max" in new_observation["daily"].keys():
        new_observation["daily"].pop("temperature_2m_max")
        new_observation["daily"].pop("temperature_2m_min")
        new_observation["daily"].pop("temperature_2m_mean")
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
                    "Subgoal": clean_observation( result )
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

class weather_toolkits:
    def __init__(self, init_config=None):
        if init_config is not None:
            if "current_date" in init_config.keys():
                self.current_date = init_config["current_date"]
            if "current_location" in init_config.keys():
                self.current_location = init_config["current_location"]

    @log_path
    def get_user_current_date(self):
        return True, self.current_date

    @log_path
    def get_user_current_location(self):
        return True, self.current_location

    @log_path
    def get_historical_temp(self, latitude=None, longitude=None, start_date=None, end_date=None, is_historical=True):
        if is_historical is True:
            # make sure that start_date and end_date are fewer than current_date
            if start_date is not None:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if start_date_obj >= current_date_obj:
                    return False, "Error: start_date should be earlier than current_date"
            if end_date is not None:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if end_date_obj >= current_date_obj:
                    return False, "Error: end_date should be earlier than current_date"
        
        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."

        def _clean(response):
            if "elevation" in response.keys():
                response.pop("elevation")
            if "generationtime_ms" in response.keys():
                response.pop("generationtime_ms")
            if "timezone" in response.keys():
                response.pop("timezone")
            if "timezone_abbreviation" in response.keys():
                response.pop("timezone_abbreviation")
            if "utc_offset_seconds" in response.keys():
                response.pop("utc_offset_seconds")
            # if "daily_units" in response.keys():
                # response.pop("daily_units")
            return response

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT",   # Use default timezone
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean"
        }
        response = requests.get(URLS["historical_weather"], params=params)
        if response.status_code == 200:
            return True, _clean( response.json() )
        else:
            return False, response.text

    @log_path
    def get_historical_rain(self, latitude=None, longitude=None, start_date=None, end_date=None, is_historical=True):
        if is_historical is True:
            # make sure that start_date and end_date are fewer than current_date
            if start_date is not None:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if start_date_obj >= current_date_obj:
                    return False, "Error: start_date should be earlier than current_date"
            if end_date is not None:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if end_date_obj >= current_date_obj:
                    return False, "Error: end_date should be earlier than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."

        def _clean(response):
            if "elevation" in response.keys():
                response.pop("elevation")
            if "generationtime_ms" in response.keys():
                response.pop("generationtime_ms")
            if "timezone" in response.keys():
                response.pop("timezone")
            if "timezone_abbreviation" in response.keys():
                response.pop("timezone_abbreviation")
            if "utc_offset_seconds" in response.keys():
                response.pop("utc_offset_seconds")
            return response

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT",   # Use default timezone
            "daily": "rain_sum"
        } 
        response = requests.get(URLS["historical_weather"], params=params)
        if response.status_code == 200:
            return True, _clean( response.json() )
        else:
            return False, response.text

    @log_path
    def get_historical_snow(self, latitude=None, longitude=None, start_date=None, end_date=None, is_historical=True):
        if is_historical is True:
            # make sure that start_date and end_date are fewer than current_date
            if start_date is not None:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if start_date_obj >= current_date_obj:
                    return False, "Error: start_date should be earlier than current_date"
            if end_date is not None:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if end_date_obj >= current_date_obj:
                    return False, "Error: end_date should be earlier than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."

        def _clean(response):
            if "elevation" in response.keys():
                response.pop("elevation")
            if "generationtime_ms" in response.keys():
                response.pop("generationtime_ms")
            if "timezone" in response.keys():
                response.pop("timezone")
            if "timezone_abbreviation" in response.keys():
                response.pop("timezone_abbreviation")
            if "utc_offset_seconds" in response.keys():
                response.pop("utc_offset_seconds")
            return response

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT",   # Use default timezone
            "daily": "snowfall_sum"
        } 
        response = requests.get(URLS["historical_weather"], params=params)
        if response.status_code == 200:
            return True, _clean( response.json() )
        else:
            return False, response.text

    @log_path
    def get_snow_forecast(self,
                          latitude=None,
                          longitude=None,
                          start_date=None,
                          end_date=None):
        # make sure that start_date and end_date are later than current_date
        if start_date is not None:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if start_date_obj <= current_date_obj:
                return False, "Error: start_date should be later than current_date"
        if end_date is not None:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if end_date_obj <= current_date_obj:
                return False, "Error: end_date should be later than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."
 
        success, response = self.get_historical_snow(latitude=latitude,
                                            longitude=longitude,
                                            start_date=start_date,
                                            end_date=end_date,
                                            is_historical=False)
        return success, response

    @log_path
    def get_current_snow(self,
                         latitude=None,
                         longitude=None,
                         current_date=None):
        success, response = self.get_historical_snow(latitude=latitude,
                                            longitude=longitude,
                                            start_date=current_date,
                                            end_date=current_date,
                                            is_historical=False
                                            )
        return success, response

    @log_path
    def get_current_temp(self, latitude=None, longitude=None, current_date=None):
        # We use get_historical_weather to get current weather
        success, response = self.get_historical_temp(latitude=latitude,
                                               longitude=longitude,
                                               start_date=current_date,
                                               end_date=current_date,
                                               is_historical=False
                                               )
        return success, response

    @log_path
    def get_latitude_longitude(self, name=None):
        def _clean(response):
            for item in response["results"]:
                if "elevation" in item.keys():
                    item.pop("elevation")
                if "feature_code" in item.keys():
                    item.pop("feature_code")
                if "country_code" in item.keys():
                    item.pop("country")
                if "country_id" in item.keys():
                    item.pop("country_id")
                if "admin1_id" in item.keys():
                    item.pop("admin1_id")
                if "timezone" in item.keys():
                    item.pop("timezone")
                if "population" in item.keys():
                    item.pop("population")
                if "postcodes" in item.keys():
                    item.pop("postcodes")
                for key in list(item.keys()):
                    if key.endswith("id"):
                        item.pop(key)
                for key in list(item.keys()):
                    if "admin" in key:
                        item.pop(key)
            if "generationtime_ms" in response.keys():
                response.pop("generationtime_ms")
            return response

        params = {
            "name": name,
            "count": 3,
            "language": "en",
            "format": "json"
        }

        response = requests.get(URLS["geocoding"], params=params)
        if response.status_code == 200:
            return True, _clean( response.json() )
        else:
            return False, response.text
    
    @log_path
    def get_air_quality(slef, latitude=None, longitude=None):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "european_aqi_pm2_5"
            # "hourly": hourly
        }
        response = requests.get(URLS["air_quality"], params=params)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text

    @log_path
    def get_elevation(self, latitude=None, longitude=None):
        params = {
            "latitude": latitude,
            "longitude": longitude
        }
        response = requests.get(URLS["elevation"], params=params)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text

    @log_path
    def get_temp_forecast(self,
                             latitude=None,
                             longitude=None,
                             start_date=None,
                             end_date=None):
        # make sure that start_date and end_date are later than current_date
        if start_date is not None:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if start_date_obj <= current_date_obj:
                return False, "Error: start_date should be later than current_date"
        if end_date is not None:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if end_date_obj <= current_date_obj:
                return False, "Error: end_date should be later than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."

        success, response = self.get_historical_temp(latitude=latitude,
                                                longitude=longitude,
                                                start_date=start_date,
                                                end_date=end_date,
                                                is_historical=False)
        return success, response

    @log_path
    def get_rain_forecast(self,
                          latitude=None,
                          longitude=None,
                          start_date=None,
                          end_date=None):
        # make sure that start_date and end_date are later than current_date
        if start_date is not None:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if start_date_obj <= current_date_obj:
                return False, "Error: start_date should be later than current_date"
        if end_date is not None:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
            if end_date_obj <= current_date_obj:
                return False, "Error: end_date should be later than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."
     
        success, response = self.get_historical_rain(latitude=latitude,
                                                longitude=longitude,
                                                start_date=start_date,
                                                end_date=end_date,
                                                is_historical=False)
        return success, response

    @log_path
    def get_current_rain(self,
                         latitude=None,
                         longitude=None,
                         current_date=None):
        success, response = self.get_historical_rain(latitude=latitude,
                                            longitude=longitude,
                                            start_date=current_date,
                                            end_date=current_date,
                                            is_historical=False
                                            )
        return success, response

    @log_path
    def get_distance(self, latitude1=None, longitude1=None, latitude2=None, 
                     longitude2=None):
        coord1 = (latitude1, longitude1)
        coord2 = (latitude2, longitude2)
        distance = geodesic(coord1, coord2).km
        return True, distance
    
    @log_path
    def get_historical_air_quality_index(self,
                           latitude=None,
                           longitude=None,
                           start_date=None,
                           end_date=None,
                           is_historical=True):
        if is_historical is True:
            # make sure that start_date and end_date are fewer than current_date
            if start_date is not None:
                start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if start_date_obj >= current_date_obj:
                    return False, "Error: start_date should be earlier than current_date"
            if end_date is not None:
                end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                current_date_obj = datetime.strptime(self.current_date, "%Y-%m-%d")
                if end_date_obj >= current_date_obj:
                    return False, "Error: end_date should be earlier than current_date"

        if is_within_30_days(start_date, end_date) is False:
            return False, "Error: Sorry, at present, we support a maximum time span of 30 days between start_date and end_date in a single query. Your input exceeds this range. You can split your current query into multiple sub-queries that meet our criteria."

        def _clean(response):
            if "elevation" in response.keys():
                response.pop("elevation")
            if "generationtime_ms" in response.keys():
                response.pop("generationtime_ms")
            if "timezone" in response.keys():
                response.pop("timezone")
            if "timezone_abbreviation" in response.keys():
                response.pop("timezone_abbreviation")
            if "utc_offset_seconds" in response.keys():
                response.pop("utc_offset_seconds")
            return response
        
        def _gather_data(response):
            new_response = {
                "latitude": response["latitude"],
                "longitude": response["longitude"],
                "daily_units": response["hourly_units"],
                "daily": response["hourly"]
            }

            # filter 12:00 data as daily data
            num_days = len(new_response["daily"]["time"]) // 24
            european_aqi_pm2_5 = []
            for i in range(num_days):
                european_aqi_pm2_5.append( new_response["daily"]["european_aqi_pm2_5"][24*i+12] )
            new_response["daily"]["european_aqi_pm2_5"] = european_aqi_pm2_5
            time = []
            for i in range(num_days):
                time.append( new_response["daily"]["time"][24*i+12] )
            new_response["daily"]["time"] = time
            return new_response

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT",   # Use default timezone
            "hourly": "european_aqi_pm2_5"
        }
        response = requests.get(URLS["air_quality"], params=params)
        if response.status_code == 200:
            response = _clean( response.json() )
            response = _gather_data( response )
            return True, response
        else:
            return False, response.text

    @log_path
    def get_current_air_quality_index(self,
                           latitude=None,
                           longitude=None,
                           current_date=None):
        success, response = self.get_historical_air_quality_index(latitude=latitude,
                                                         longitude=longitude,
                                                         start_date=current_date,
                                                         end_date=current_date,
                                                         is_historical=False)
        return success, response

    @log_path
    def get_air_quality_level(self, air_quality_index):
        response = None
        if air_quality_index <= 20:
            response = "good"
        elif 21 < air_quality_index <= 40:
            response = "fair"
        elif 41 < air_quality_index <= 60:
            response = "moderate"
        elif 61 < air_quality_index <= 80:
            response = "poor"
        elif 81 < air_quality_index <= 100:
            response = "very poor"
        else:
            response = "extremely poor"
        return True, response
    
    @log_path
    def convert_zipcode_to_address(self, zipcode):
        response = requests.get(URLS["zipcode"].format(zipcode=zipcode))
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.text
    
    @log_path
    def finish(self, answer):
        if type(answer) == list:
            answer = sorted(answer)
        return True, answer

    # @log_path

if __name__ == "__main__":
    init_config = {
        "current_date": "2023-01-01"
    }
    tool = weather_toolkits(init_config=init_config)
    
    # print( tool.get_temp_forecast(latitude=40.699997,
                                #   longitude= ) )
    # print( tool.get_weather_forecast(latitude="52.52", longitude="13.41") )
    # print( tool.get_weather_forecast(latitude="52.52", longitude="13.41") )
    # print( tool.get_eveluation(latitude="52.52", longitude="13.41") )
    # pprint( tool.get_air_quality(latitude="40.7128", longitude="-74.0060") )
    # pprint(tool.get_geocoding(name="New York"))
    print(tool.get_historical_temp(latitude=31.22222, longitude=121.45806, start_date="2015-01-01", end_date="2015-03-01"))
    # pprint( tool.get_historical_air(latitude=52.52, longitude=13.41, start_date="2022-11-01", end_date="2022-11-30") )
    # pprint( tool.convert_zipcode_to_address("84323") )
    # print( tool.get_geocoding(name="Shanghai") )