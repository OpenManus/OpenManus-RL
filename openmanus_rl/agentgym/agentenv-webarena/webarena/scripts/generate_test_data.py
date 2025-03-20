"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json

from browser_env.env_config import *


def main() -> None:
    with open("config_files/test.raw.json", "r") as f:
        raw = f.read()
    raw = raw.replace("__GITLAB__", GITLAB)
    raw = raw.replace("__REDDIT__", REDDIT)
    raw = raw.replace("__SHOPPING__", SHOPPING)
    raw = raw.replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
    raw = raw.replace("__WIKIPEDIA__", WIKIPEDIA)
    raw = raw.replace("__MAP__", MAP)
    with open("config_files/test.json", "w") as f:
        f.write(raw)
    # split to multiple files
    data = json.loads(raw)
    for idx, item in enumerate(data):
        with open(f"config_files/{idx}.json", "w") as f:
            json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()
