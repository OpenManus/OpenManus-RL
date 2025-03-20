import glob
import importlib
import json
import os


# use the current directory as the root
def run() -> None:
    """Convert all python files in agent/prompts to json files in agent/prompts/jsons

    Python files are easiser to edit
    """
    for p_file in glob.glob(f"agent/prompts/raw/*.py"):
        # import the file as a module
        base_name = os.path.basename(p_file).replace(".py", "")
        module = importlib.import_module(f"agent.prompts.raw.{base_name}")
        prompt = module.prompt
        # save the prompt as a json file
        os.makedirs("agent/prompts/jsons", exist_ok=True)
        with open(f"agent/prompts/jsons/{base_name}.json", "w+") as f:
            json.dump(prompt, f, indent=2)
    print(f"Done convert python files to json")


if __name__ == "__main__":
    run()
