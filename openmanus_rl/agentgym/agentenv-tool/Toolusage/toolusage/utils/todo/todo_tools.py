import os
import json
import requests
from typing import List, Dict, Any, Union
from copy import deepcopy
from dotenv import load_dotenv

load_dotenv()

URLS = {
    "get_projects": "https://api.todoist.com/rest/v2/projects",
    "add_projects": "https://api.todoist.com/rest/v2/projects",
    "update_project": "https://api.todoist.com/rest/v2/projects/{project_id}",
    "get_sections": "https://api.todoist.com/rest/v2/sections",
    "add_sections": "https://api.todoist.com/rest/v2/sections",
    "get_tasks": "https://api.todoist.com/rest/v2/tasks",
    "add_tasks": "https://api.todoist.com/rest/v2/tasks",
    "update_tasks": "https://api.todoist.com/rest/v2/tasks/{task_id}",
    "complete_tasks": "https://api.todoist.com/rest/v2/tasks/{task_id}/close",
    "delete_projects": "https://api.todoist.com/rest/v2/projects/{project_id}",
    "get_task_by_id": "https://api.todoist.com/rest/v2/tasks/{task_id}",
    "get_project_by_id": "https://api.todoist.com/rest/v2/projects/{project_id}",
    "delete_task": "https://api.todoist.com/rest/v2/tasks/{task_id}"
}

GET_HEADERS = {
    "Authorization" : "Bearer {}".format(os.environ["TODO_KEY"] if "TODO_KEY" in os.environ.keys() else "")
}

POST_HEADERS = {
    "Authorization" : "Bearer {}".format(os.environ["TODO_KEY"] if "TODO_KEY" in os.environ.keys() else "")
}

def clean_observation(
                        observation : Union[ List[Dict[str, Any]], Dict[str, Any] ]
                    ):
    # remove all "id" in observation
    new_observation = deepcopy(observation)
    if isinstance(new_observation, list):
        for item in new_observation:
            if isinstance(item, dict):
                item.pop("id")
    elif isinstance(new_observation, dict):
        if "id" in new_observation.keys():
            new_observation.pop("id")
        if "task" in new_observation.keys():
            new_observation["task"].pop("id")
        if "project" in new_observation.keys():
            new_observation["project"].pop("id")
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

class todo_toolkits:
    def __init__(self, init_config=None):
        # remove all projects and upload the initial project
        # 1. remove all projects
        print("Removing all projects from account")
        _, all_projects = self.get_projects()
        for project in all_projects:
            self.delete_project(project_id=project["id"])

        # 2. upload the initial project
        print("Initializing the project")
        self.initial_project()
        print("Initialization completed")

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

    def initial_project(self):
        with open("{}/toolusage/utils/todo/projects.json".format(os.environ["PROJECT_PATH"]), "r") as f:
            projects = json.load(f)
        for project in projects:
            _, returned_project = self.add_project(project={"name": project["name"]})
            project_id = returned_project["id"]
            for task in project["tasks"]:
                self.add_task(project_id=project_id,
                                content=task["content"],
                                description=task["description"],
                                due_date=task["due_date"],
                                duration=task["duration"],
                                duration_unit=task["duration_unit"],
                                priority=task["priority"],
                                # labels=task["labels"]
                                )

    @log_path
    def get_projects(self):
        response = requests.get(URLS["get_projects"], headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            for project in response:
                self._clean_project(project)
            # remove project "Inbox"
            response = [project for project in response if project["name"] != "Inbox"]
            return True, response
        else:
            return False, response.text

    @log_path
    def add_project(self, project=None):
        params = project
        response = requests.post(URLS["add_projects"], headers=POST_HEADERS, params=params)
        if response.status_code == 200:
            response = response.json()
            project = self._clean_project(response)
            return True, project
        else:
            return False, response.text

    @log_path
    def update_project(self, project_id=None, is_favorite=None):
        params = {}
        if is_favorite is not None:
            params["is_favorite"] = is_favorite
        response = requests.post(URLS["update_project"].format(project_id=project_id), headers=POST_HEADERS, params=params)
        if response.status_code == 200:
            response = response.json()
            project = self._clean_project(response)
            returned_object = {}
            returned_object["message"] = "Update the project successfully."
            returned_object["project"] = project
            return True, returned_object
            return f"Update project - {project['name']} successfully."
        else:
            return False, response.text
    
    @log_path
    def get_tasks(self, project_id=None):
        params = {
            "project_id": project_id
        }
        response = requests.get(URLS["get_tasks"], headers=GET_HEADERS, params=params)
        if response.status_code == 200:
            response = response.json()
            for task in response:
                self._clean_task(task)
            return True, response
        else:
            return False, response.text

    @log_path
    def get_task_description(self, task_id=None):
        response = requests.get(URLS["get_task_by_id"].format(task_id=task_id), headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            return_data = {
                "id": response["id"],
                "content": response["content"],
                "description": response["description"]
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def get_task_duration(self, task_id=None):
        response = requests.get(URLS["get_task_by_id"].format(task_id=task_id), headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            return_data = {
                "id": response["id"],
                "content": response["content"],
                "duration": str(response["duration"]["amount"]) + "(" + response["duration"]["unit"] + ")" 
            }
            return True, return_data
        else:
            return False, response.text

    @log_path
    def add_task(self,
                 project_id=None,
                 content=None,
                 description=None,
                 due_date=None,
                 duration=None,
                 duration_unit=None,
                 priority=None,
                 labels=None
                 ):
        params = {
            "project_id": project_id,
            "content": content,
            "description": description,
            "due_date": due_date,
            "duration": duration,
            "duration_unit": duration_unit,
            "priority": priority
            # "labels": labels
        }
        response = requests.post(URLS["add_tasks"], headers=POST_HEADERS, params=params)
        if response.status_code == 200:
            response = response.json()
            response = self._clean_task(response)
            return True, response
        else:
            return False, response.text

    @log_path
    def complete_task(self, task_id=None):
        # First, we should whether this task is exists
        response = requests.get(URLS["get_task_by_id"].format(task_id=task_id), headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            task = self._clean_task(response)
        else:
            return False, f"Task {task_id} is not exists."

        # Second, we should complete this task 
        response = requests.post(URLS["complete_tasks"].format(task_id=task_id), headers=POST_HEADERS)
        if response.status_code == 204:
            return_object = {}
            return_object["message"] = "Complete task successfully."
            task.pop("is_completed")
            return_object["task"] = task
            return True, return_object
        else:
            return False, response.text
    
    @log_path
    def update_task(self, task_id=None, due_date=None):
        params = {
            "due_date": due_date
        }
        response = requests.post(URLS["update_tasks"].format(task_id=task_id), headers=POST_HEADERS, params=params)
        if response.status_code == 200:
            task = response.json()
            task = self._clean_task(task)
            returned_object = {}
            returned_object["message"] = "Update the task successfully"
            returned_object["task"] = task
            return True, returned_object
        else:
            return False, response.text

    @log_path
    def delete_project(self, project_id=None):
        # First, we should whether this project is exists
        response = requests.get(URLS["get_project_by_id"].format(project_id=project_id), headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            project = self._clean_project(response)
        else:
            return False, f"Project {project_id} is not exists."
        
        # Second, we should delete this project
        response = requests.delete(URLS["delete_projects"].format(project_id=project_id), headers=POST_HEADERS)
        # return response.json()
        if response.status_code == 204:
            return True, project
        else:
            return False, response.text
    
    @log_path
    def delete_task(self, task_id=None):
        # First, we should whether this task is exists
        response = requests.get(URLS["get_task_by_id"].format(task_id=task_id), headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            task = self._clean_task(response)
        else:
            return False, f"Task {task_id} is not exists."

        # Second, we should delete this task
        response = requests.delete(URLS["delete_task"].format(task_id=task_id), headers=POST_HEADERS)
        if response.status_code == 204:
            returned_object = {}
            returned_object["message"] = "Delete the task successfully."
            returned_object["task"] = task
            return True, returned_object
        else:
            return False, response.text

    def _clean_task(self, task):
        task.pop("assigner_id")
        task.pop("assignee_id")
        task.pop("section_id")
        task.pop("parent_id")
        task.pop("labels")
        task.pop("comment_count")
        task.pop("creator_id")
        task.pop("created_at")
        task.pop("url")
        task.pop("project_id")
        task["due_date"] = task["due"]["date"]
        task.pop("due")
        task.pop("description")
        task.pop("duration")
        # task["duration"] = str(task["duration"]["amount"]) + "(" + task["duration"]["unit"] + ")"
        return task       

    def _clean_project(self, project):
        project.pop("comment_count")
        project.pop("is_shared")
        project.pop("is_inbox_project")
        project.pop("is_team_inbox")
        project.pop("url")
        project.pop("view_style")
        project.pop("parent_id")
        return project
    
    def _get_all_projects(self):
        response = requests.get(URLS["get_projects"], headers=GET_HEADERS)
        if response.status_code == 200:
            response = response.json()
            for project in response:
                self._clean_project(project)
            # remove project "Inbox"
            response = [project for project in response if project["name"] != "Inbox"]
            return True, response
        else:
            return False, response.text

    def _get_all_tasks(self):
        # response = requests.get(URLS["get_tasks"], headers=GET_HEADERS)
        # get project first 
        tasks = []
        _, projects = self._get_all_projects()
        for project in projects:
            tasks.extend(self.get_tasks(project_id=project["id"])[1] )

        return True, tasks

    @log_path
    def finish(self, answer):
        if type(answer) == list:
            answer = sorted(answer)
        return True, answer
    
if __name__ == "__main__":
    tool = todo_toolkits()