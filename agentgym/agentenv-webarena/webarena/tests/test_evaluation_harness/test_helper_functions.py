import json
import os
from pathlib import Path

from browser_env import ScriptBrowserEnv
from browser_env.env_config import *
from evaluation_harness.helper_functions import (
    gitlab_get_project_memeber_role,
)

HEADLESS = True
config_file_folder = "tests/test_evaluation_harness/configs"


def test_gitlab_get_project_memeber_role(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/tmp_config.json"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/gitlab_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{GITLAB}/primer/design/-/project_members")
    role1 = gitlab_get_project_memeber_role(env.page, "byteblaze")
    assert role1 == "Developer"
    role2 = gitlab_get_project_memeber_role(env.page, "primer")
    assert role2 == "Owner"

    # remove tmp config file
    os.remove(config_file)
