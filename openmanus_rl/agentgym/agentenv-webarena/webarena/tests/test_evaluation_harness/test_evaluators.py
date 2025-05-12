import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Any

import pytest
from py import test

from agent import Agent, TeacherForcingAgent
from browser_env import ActionTypes, ScriptBrowserEnv
from browser_env.env_config import *
from evaluation_harness import (
    HTMLContentEvaluator,
    StringEvaluator,
    URLEvaluator,
)
from evaluation_harness.evaluators import EvaluatorComb

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
HEADLESS = True
config_file_folder = "tests/test_evaluation_harness/configs"


def tf_roll_out(
    agent: Agent, env: ScriptBrowserEnv, config_file: str
) -> list[Any]:
    """Roll out the agent using teacher forcing actions"""
    obs, state_info = env.reset(options={"config_file": config_file})

    trajectory: list[Any] = [{"observation": obs, "info": state_info}]
    while True:
        action = agent.next_action(
            trajectory=trajectory, intent="", meta_data={}
        )
        trajectory.append(action)
        if action["action_type"] == ActionTypes.STOP:
            break

        # preceed to next action
        obs, reward, terminated, truncated, info = env.step(action)
        state_info = {"observation": obs, "info": info}
        trajectory.append(state_info)

    return trajectory


def test_string_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/string_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.stop("The date is 1985/04/18")"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = StringEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )

    assert score == 1.0


def test_string_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/string_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.stop("The date is 1936/04/18")"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = StringEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )

    assert score == 0.0


def test_url_exact_match_success(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/url_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://www.google.com/")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = URLEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


def test_url_exact_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/url_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("{GITLAB}")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = URLEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    print(env.page.url)
    assert score == 0.0


def test_html_content_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_exact_match.json"

    # randomly sample a string
    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


def test_html_content_match_fail(script_browser_env: ScriptBrowserEnv) -> None:
    config_file = f"{config_file_folder}/html_content_exact_match.json"

    # randomly sample a string
    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = """page.goto("https://www.google.com/")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 0.0


def test_html_content_element_match_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_element_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello World")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


def test_html_content_element_match_fail(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_element_exact_match.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 0.0


def test_html_content_url_comb_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/html_content_url_comb.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("https://russmaxdesign.github.io/exercise/")
    page.get_by_label("Full name").fill("Hello World")
    page.get_by_label("Email").click()
    page.get_by_label("Email").fill("alexisxy@hotmail.com")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env

    trajectory = tf_roll_out(agent, env, config_file)

    evaluators = EvaluatorComb([URLEvaluator(), HTMLContentEvaluator()])
    score = evaluators(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Won't work using the demo sites"
)
def test_func_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_eval_success.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS, reason="Won't work using the demo sites"
)
def test_func_fail(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_eval_fail.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 0.0


def test_func_url_func_last_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_url_func_1.json"

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.goto("{REDDIT}/f/wallstreetbets/50431/-/comment/676875")
    page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, config_file)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, config_file, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0


def test_func_url_func_page_success(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    config_file = f"{config_file_folder}/func_url_func_2.json"

    # change the URL placeholder with the concrete URL
    with open(config_file, "r") as f:
        configs = json.load(f)
        configs["eval"]["program_html"][0]["url"] = configs["eval"][
            "program_html"
        ][0]["url"].replace("__GITLAB__", GITLAB)
        configs["eval"]["program_html"][1]["url"] = configs["eval"][
            "program_html"
        ][1]["url"].replace("__GITLAB__", GITLAB)
    tmp_config = config_file.replace(".json", ".tmp.json")
    with open(tmp_config, "w+") as f:
        json.dump(configs, f, indent=4)

    agent = TeacherForcingAgent()
    agent.set_action_set_tag(tag="playwright")
    action_seq = f"""page.stop()"""
    agent.set_actions(action_seq)

    env = script_browser_env
    trajectory = tf_roll_out(agent, env, tmp_config)

    evalutor = HTMLContentEvaluator()
    score = evalutor(
        trajectory, tmp_config, env.page, env.get_page_client(env.page)
    )
    assert score == 1.0
    os.remove(tmp_config)
