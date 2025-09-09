# Grounding DINO Object Detection Tool
# https://huggingface.co/IDEA-Research/grounding-dino

import os
import time

from openmanus_rl.tools.base import BaseTool
from PIL import Image, ImageOps

import os
# Suppress stderr by redirecting it to /dev/null
import sys
import re
import base64
import requests
sys.stderr = open(os.devnull, 'w')


class Advanced_Object_Detector_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Advanced_Object_Detector_Tool",
            tool_description="A tool that detects objects in an image using the Grounding DINO-X model and saves individual object images with empty padding.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "labels": "list - A list of object labels to detect.",
                "threshold": "float - The confidence threshold for detection (default: 0.35).",
                "padding": "int - The number of pixels to add as empty padding around detected objects (default: 20)."
            },
            output_type="list - A list of detected objects with their scores, bounding boxes, and saved image paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", labels=["baseball", "basket"])',
                    "description": "Detect baseball and basket in an image, save the detected objects with default empty padding, and return their paths."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", labels=["car", "person"], threshold=0.5, model_size="base", padding=15)',
                    "description": "Detect car and person in an image using the base model, save the detected objects with 15 pixels of empty padding, and return their paths."
                }
            ],
            user_metadata={
                "limitation": "The model may not always detect objects accurately, and its performance can vary depending on the input image and the associated labels. It typically struggles with detecting small objects, objects that are uncommon, or objects with limited or specific attributes. For improved accuracy or better detection in certain situations, consider using supplementary tools or image processing techniques to provide additional information for verification."
            }
        )
        self.DINO_KEY = os.environ.get("DINO_KEY")

    def preprocess_caption(self, caption):
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def build_tool(self, threshold=0.35):

        params_dict = {
                        'headers': {
                                    "Content-Type": "application/json",
                                    "Token"       : self.DINO_KEY
                                    },
                        'body':{
                                    "image"  : None,
                                    "prompts": [
                                        {"type": "text", "text": None},
                                    ],
                                    "bbox_threshold": threshold 
                                }

                      }
        return params_dict


    def save_detected_object(self, image, box, image_name, label, index, padding):
        object_image = image.crop(box)
        padded_image = ImageOps.expand(object_image, border=padding, fill='white')
        
        filename = f"{image_name}_{label}_{index}.png"
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        
        padded_image.save(save_path)
        return save_path

    def execute(self, image, labels, threshold=0.35, padding=20, max_retries=10, retry_delay=5):
        retry_count = 0
        params = self.build_tool(threshold)

        def process_image(input_str):

            def image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            # Define common image file extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.webp'}

            # Check if it is a URL
            url_pattern = re.compile(r'^(http|https|ftp)://')
            if url_pattern.match(input_str):
                if input_str.lower().endswith(tuple(image_extensions)):
                    return input_str
                return input_str

            # Check if it is a file path
            _, ext = os.path.splitext(input_str)
            if ext.lower() in image_extensions:
                image_base64 = image_to_base64(input_str)
                return f'data:image/png;base64,{image_base64}'
            return None

        if len(labels) < 1:
            preprocessed_prompt = '<prompt_free>'
        else:
            preprocessed_prompt = ''
            for label in labels:
                preprocessed_prompt += self.preprocess_caption(label)


        body = params['body']
        body['image'] = process_image(image)
        body['prompts'] =  [{"type": "text", "text": preprocessed_prompt}]

        # send request
        resp = requests.post(
            'https://api.deepdataspace.com/tasks/dinox',
            json=body,
            headers=params['headers']
        )

        if resp.status_code == 200:
            json_resp = resp.json()
            print(json_resp)

            # get task_uuid
            task_uuid = json_resp["data"]["task_uuid"]
            print(f'task_uuid:{task_uuid}')

            # poll get task result
            while retry_count < max_retries:
                resp = requests.get(f'https://api.deepdataspace.com/task_statuses/{task_uuid}', headers=params['headers'])
                

                if resp.status_code != 200:
                    break
                json_resp = resp.json()

                if json_resp["data"]["status"] not in ["waiting", "running"]:
                    break
                time.sleep(1)#retry_delay)
                retry_count += 1

            if json_resp["data"]["status"] == "failed":
                print(f'failed resp: {json_resp}')
            elif json_resp["data"]["status"] == "success":
                # print(f'success resp: {json_resp}')
                formatted_results = []
                original_image = Image.open(image)
                image_name = os.path.splitext(os.path.basename(image))[0]
                
                object_counts = {}

                for result in json_resp['data']['result']['objects']:
                    box = tuple(result["bbox"])
                    try:
                        box = [int(x) for x in box]
                    except:
                        continue
                    label = result["category"]
                    score = round(result["score"], 2)
                    if label.endswith("."):
                        label = label[:-1]
                    
                    object_counts[label] = object_counts.get(label, 0) + 1
                    index = object_counts[label]

                    save_path = self.save_detected_object(original_image, box, image_name, label, index, padding)
            
                    formatted_results.append({
                        "label": label,
                        "confidence score": score,
                        "box": box,
                        "saved_image_path": save_path
                    })

                return formatted_results
            else:
                print(f'get task resp: {resp.status_code} - {resp.text}')
        else:
            print(f'Error: {resp.status_code} - {resp.text}')
        
        print(f"Failed to detect objects after {max_retries} attempts.")
        return []

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/advanced_object_detector
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Object_Detector_Tool
    tool = Advanced_Object_Detector_Tool()
    tool.set_custom_output_dir("detected_objects")

    # Get tool metadata
    metadata = tool.get_metadata()
    # print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/baseball.png"
    image_path = os.path.join(script_dir, relative_image_path)

    import json

    # Execute the tool
    try:
        execution = tool.execute(image=image_path, labels=["baseball", "basket"], padding=20)
        print(json.dumps(execution, indent=4))
        print("Detected Objects:")
        for obj in execution:
            print(f"Detected {obj['label']} with confidence {obj['confidence score']}")
            print(f"Bounding box: {obj['box']}")
            print(f"Saved image (with padding): {obj['saved_image_path']}")
            print()
    except ValueError as e: 
        print(f"Execution failed: {e}")

    print("Done!")
