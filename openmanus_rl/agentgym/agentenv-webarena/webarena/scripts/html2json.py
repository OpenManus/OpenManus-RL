import argparse
import base64
import glob
import json
import os
from collections import defaultdict
from typing import Any

from bs4 import BeautifulSoup


def main(result_folder: str, config_json: str) -> None:
    all_data = {}
    template_to_id: dict[str, Any] = defaultdict(lambda: len(template_to_id))

    with open(config_json, "r") as f:
        data_configs = json.load(f)
        data_configs = {int(item["task_id"]): item for item in data_configs}
        for k, v in data_configs.items():
            v.pop("require_login")
            v.pop("storage_state")
            v.pop("start_url")
            v.pop("geolocation")
            v.pop("require_reset")
            v.pop("intent_template_id")
            v["intent_template_id"] = template_to_id[v["intent_template"]]
            v["eval_types"] = v["eval"].pop("eval_types")
            if v["eval"]["reference_answers"]:
                v["reference_answers"] = v["eval"].pop("reference_answers")
            if v["eval"]["reference_url"]:
                v["reference_url"] = v["eval"].pop("reference_url")
            v.pop("eval")
            if v.get("reference_answers", {}).get("exact_match", "") == "N/A":
                v["achievable"] = False
            else:
                v["achievable"] = True

    with open(f"{result_folder}/merged_log.txt", "r") as f:
        results = {}
        for line in f:
            if "[Result]" in line:
                id = line.strip().split(".")[-2].split("/")[-1]
                results[int(id)] = True if "(PASS)" in line else False

    files = list(glob.glob(f"{result_folder}/render_*.html"))
    files = [x for x in files if os.path.exists(x)]
    print(f"Total number of files: {len(files)}")

    for render_file in files:
        task_id = int(render_file.split("_")[-1].split(".")[0])
        with open(render_file, "r") as f:
            try:
                content = f.read()
                soup = BeautifulSoup(content, "html.parser")
                observations = [
                    obv.find("pre").text
                    for obv in soup.find_all("div", {"class": "state_obv"})
                ]
                base64_images = [
                    img["src"].split(",")[1] for img in soup.find_all("img")
                ]
                image_observations = []
                # save image to file and change the value to be path
                image_folder = f"images/{os.path.basename(result_folder)}"
                os.makedirs(image_folder, exist_ok=True)
                for i, image in enumerate(base64_images):
                    image_data = base64.b64decode(image)
                    filename = f"{image_folder}/image_{task_id}_{i}.png"
                    with open(filename, "wb") as f:  # type: ignore[assignment]
                        f.write(image_data)  # type: ignore[arg-type]
                    image_observations.append(filename)
                urls = [
                    url.get_text()
                    for url in soup.find_all("h3", {"class": "url"})
                ]
                actions = [
                    action.get_text()
                    for action in soup.find_all(
                        "div", {"class": "raw_parsed_prediction"}
                    )
                ]
                parsed_actions = [
                    action.get_text()
                    for action in soup.find_all(
                        "div", {"class": "parsed_action"}
                    )
                ]
                # fill action with parsed action if action is empty
                for i in range(len(actions)):
                    if actions[i] == "":
                        actions[i] = parsed_actions[i]

                messages = []
                for o, u, a, image in zip(
                    observations, urls, actions, image_observations
                ):
                    messages.append(
                        {
                            "user": f"{u}\n\nobservation:\n{o}",
                            "image": image,
                        }
                    )
                    messages.append({"assistant": a})

                all_data[f"example_{task_id}"] = {
                    **data_configs[task_id],
                    "messages": messages,
                    "success": results.get(task_id, False),
                }

            except Exception as e:
                print(e)
                print(f"Error in {render_file}")

    with open(f"{result_folder}/json_dump.json", "w+") as f:
        json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str)
    parser.add_argument(
        "--config_json", type=str, default="config_files/test.raw.json"
    )
    args = parser.parse_args()
    main(args.result_folder, args.config_json)
