# Grounding DINO Object Detection Tool
# https://huggingface.co/IDEA-Research/grounding-dino

import os
import time
import torch
from transformers import pipeline

from openmanus_rl.tools.base import BaseTool
from PIL import Image, ImageOps

import os
# If CUDA_HOME is set, print the value
print(os.environ.get('CUDA_HOME', 'CUDA_HOME is not set'))

# Suppress stderr by redirecting it to /dev/null
import sys
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")


class Object_Detector_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Object_Detector_Tool",
            tool_description="A tool that detects objects in an image using the Grounding DINO model and saves individual object images with empty padding.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "labels": "list - A list of object labels to detect.",
                "threshold": "float - The confidence threshold for detection (default: 0.35).",
                "model_size": "str - The size of the model to use ('tiny' or 'base', default: 'tiny').",
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

    def preprocess_caption(self, caption):
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def build_tool(self, model_size='tiny'):
        model_name = f"IDEA-Research/grounding-dino-{model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            pipe = pipeline(model=model_name, task="zero-shot-object-detection", device=device)
            return pipe
        except Exception as e:
            print(f"Error building the Object Detection tool: {e}")
            return None

    def save_detected_object(self, image, box, image_name, label, index, padding):
        object_image = image.crop(box)
        padded_image = ImageOps.expand(object_image, border=padding, fill='white')
        
        filename = f"{image_name}_{label}_{index}.png"
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        
        padded_image.save(save_path)
        return save_path

    def execute(self, image, labels, threshold=0.35, model_size='tiny', padding=20, max_retries=10, retry_delay=5, clear_cuda_cache=False):
        for attempt in range(max_retries):
            try:
                saved_files = []

                pipe = self.build_tool(model_size)
                if pipe is None:
                    raise ValueError("Failed to build the Object Detection tool.")
                
                preprocessed_labels = [self.preprocess_caption(label) for label in labels]
                results = pipe(image, candidate_labels=preprocessed_labels, threshold=threshold)
                
                formatted_results = []
                original_image = Image.open(image)
                image_name = os.path.splitext(os.path.basename(image))[0]
                
                object_counts = {}

                for result in results:
                    box = tuple(result["box"].values())
                    label = result["label"]
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
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory error on attempt {attempt + 1}.")
                    if clear_cuda_cache:
                        print("Clearing CUDA cache and retrying...")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Runtime error: {e}")
                    break
            except Exception as e:
                print(f"Error detecting objects: {e}")
                break
        
        print(f"Failed to detect objects after {max_retries} attempts.")
        return []

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/object_detector
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Object_Detector_Tool
    tool = Object_Detector_Tool()
    tool.set_custom_output_dir("detected_objects")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/baseball.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Execute the tool
    try:
        execution = tool.execute(image=image_path, labels=["baseball", "basket"], padding=20)
        print("Detected Objects:")
        for obj in execution:
            print(f"Detected {obj['label']} with confidence {obj['confidence score']}")
            print(f"Bounding box: {obj['box']}")
            print(f"Saved image (with padding): {obj['saved_image_path']}")
            print()
    except ValueError as e: 
        print(f"Execution failed: {e}")

    print("Done!")
