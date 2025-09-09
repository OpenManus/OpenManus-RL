import os
import cv2
from pydantic import BaseModel
from openmanus_rl.tools.base import BaseTool
from openmanus_rl.engines.factory import create_llm_engine

class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]

class Relevant_Patch_Zoomer_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o"):
        super().__init__(
            tool_name="Relevant_Patch_Zoomer_Tool",
            tool_description="A tool that analyzes an image, divides it into 5 regions (4 quarters + center), and identifies the most relevant patches based on a question. The returned patches are zoomed in by a factor of 2.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "question": "str - The question about the image content.",
            },
            output_type="dict - Contains analysis text and list of saved zoomed patch paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.jpg", question="What is the color of the car?")',
                    "description": "Analyze image and return relevant zoomed patches that show the car's color."
                }
            ],
            user_metadata = {
                "best_practices": [
                    "It might be helpful to zoom in on the image first to get a better look at the object(s).",
                    "It might be helpful if the question requires a close-up view of the object(s), symbols, texts, etc.",
                    "The tool should be used to provide a high-level analysis first, and then use other tools for fine-grained analysis. For example, you can use Relevant_Patch_Zoomer_Tool first to get a zoomed patch of specific objects, and then use Image_Captioner_Tool to describe the objects in detail."
                ]
            }
        )
        self.matching_dict = {
            "A": "top-left",
            "B": "top-right",
            "C": "bottom-left",
            "D": "bottom-right",
            "E": "center"
        }

        print(f"Initializing Patch Zoomer Tool with model: {model_string}")
        self.llm_engine = create_llm_engine(model_string=model_string, is_multimodal=True) if model_string else None
        
    def _save_patch(self, image_path, patch, save_path, zoom_factor=2):
        """Extract and save a specific patch from the image with 10% margins."""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        quarter_h = height // 2
        quarter_w = width // 2
        
        margin_h = int(quarter_h * 0.1)
        margin_w = int(quarter_w * 0.1)
        
        patch_coords = {
            'A': ((max(0, 0 - margin_w), max(0, 0 - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, quarter_h + margin_h))),
            'B': ((max(0, quarter_w - margin_w), max(0, 0 - margin_h)),
                  (min(width, width + margin_w), min(height, quarter_h + margin_h))),
            'C': ((max(0, 0 - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, height + margin_h))),
            'D': ((max(0, quarter_w - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, width + margin_w), min(height, height + margin_h))),
            'E': ((max(0, quarter_w//2 - margin_w), max(0, quarter_h//2 - margin_h)),
                  (min(width, quarter_w//2 + quarter_w + margin_w), 
                   min(height, quarter_h//2 + quarter_h + margin_h)))
        }
        
        (x1, y1), (x2, y2) = patch_coords[patch]
        patch_img = img[y1:y2, x1:x2]
        
        zoomed_patch = cv2.resize(patch_img, 
                                (patch_img.shape[1] * zoom_factor, 
                                 patch_img.shape[0] * zoom_factor), 
                                interpolation=cv2.INTER_LINEAR)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, zoomed_patch)
        return save_path

    def execute(self, image, question, zoom_factor=2):
        try:
            if not self.llm_engine:
                return "Error: LLM engine not initialized. Please provide a valid model_string."
            
            # Prepare the prompt
            prompt = f"""
Analyze this image to identify the most relevant region(s) for answering the question:

Question: {question}

The image is divided into 5 regions:
- (A) Top-left quarter
- (B) Top-right quarter
- (C) Bottom-left quarter
- (D) Bottom-right quarter
- (E) Center region (1/4 size, overlapping middle section)

Instructions:
1. First describe what you see in each of the five regions.
2. Then select the most relevant region(s) to answer the question.
3. Choose only the minimum necessary regions - avoid selecting redundant areas that show the same content. For example, if one patch contains the entire object(s), do not select another patch that only shows a part of the same object(s).


Response format:
<analysis>: Describe the image and five patches first. Then analyze the question and select the most relevant patch or list of patches.
<patch>: List of letters (A-E)
"""
            # Read image and create input data
            with open(image, 'rb') as file:
                image_bytes = file.read()
            input_data = [prompt, image_bytes]
            
            # Get response from LLM
            response = self.llm_engine(input_data, response_format=PatchZoomerResponse)
            
            # Save patches
            image_dir = os.path.dirname(image)
            image_name = os.path.splitext(os.path.basename(image))[0]
            
            # Update the return structure
            patch_info = []
            for patch in response.patch:
                patch_name = self.matching_dict[patch]
                save_path = os.path.join(self.output_dir, 
                                       f"{image_name}_{patch_name}_zoomed_{zoom_factor}x.png")
                saved_path = self._save_patch(image, patch, save_path, zoom_factor)
                save_path = os.path.abspath(saved_path)
                patch_info.append({
                    "path": save_path,
                    "description": f"The {self.matching_dict[patch]} region of the image: {image}."
                })
            
            return {
                "analysis": response.analysis,
                "patches": patch_info
            }
            
        except Exception as e:
            print(f"Error in patch zooming: {e}")
            return None

    def get_metadata(self):
        return super().get_metadata()

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/relevant_patch_zoomer
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Relevant_Patch_Zoomer_Tool
    tool = Relevant_Patch_Zoomer_Tool()
    tool.set_custom_output_dir(f"{script_dir}/zoomed_patches")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/car.png"
    image_path = os.path.join(script_dir, relative_image_path)
    question = "What is the color of the car?"

    # Execute the tool
    try:
        result = tool.execute(image=image_path, question=question)
        if result:
            print("\nDetected Patches:")
            for patch in result['patches']:
                print(f"Path: {patch['path']}")
                print(f"Description: {patch['description']}")
                print()
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
