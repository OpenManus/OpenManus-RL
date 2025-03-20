
from common.registry import registry
import json
import os

def load_environment(name, config):
    
    if name not in registry.list_environments():
        if name == "academia": from environment.academia_env import AcademiaEnv
        if name == "todo": from environment.todo_env import TodoEnv
        if name == "movie": from environment.movie_env import MovieEnv
        if name == "weather": from environment.weather_env import WeatherEnv
        if name == "sheet": from environment.sheet_env import SheetEnv

    
    env = registry.get_environment_class(name).from_config(config)

    return env

