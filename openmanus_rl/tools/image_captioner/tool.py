import os
from openmanus_rl.tools.base import BaseTool
from openmanus_rl.engines.factory import create_llm_engine

class Image_Captioner_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name="Image_Captioner_Tool",
            tool_description="A tool that generates captions for images using OpenAI's multimodal model.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "prompt": "str - The prompt to guide the image captioning (default: 'Describe this image in detail.').",
            },
            output_type="str - The generated caption for the image.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png")',
                    "description": "Generate a caption for an image using the default prompt and model."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", prompt="Explain the mood of this scene.")',
                    "description": "Generate a caption focusing on the mood using a specific prompt and model."
                }
            ],
            user_metadata = {
                "limitation": "The Image_Captioner_Tool provides general image descriptions but has limitations: 1) May make mistakes in complex scenes, counting, attribute detection, and understanding object relationships. 2) Might not generate comprehensive captions, especially for images with multiple objects or abstract concepts. 3) Performance varies with image complexity. 4) Struggles with culturally specific or domain-specific content. 5) May overlook details or misinterpret object relationships. For precise descriptions, consider: using it with other tools for context/verification, as an initial step before refinement, or in multi-step processes for ambiguity resolution. Verify critical information with specialized tools or human expertise when necessary."
            },
        )
        print(f"Initializing Image Captioner Tool with model: {model_string}")
        self.llm_engine = create_llm_engine(model_string=model_string, is_multimodal=True) if model_string else None

    def execute(self, image, prompt="Describe this image in detail."):
        try:
            if not self.llm_engine:
                return "Error: LLM engine not initialized. Please provide a valid model_string."
                
            input_data = [prompt]
            
            if image and os.path.isfile(image):
                try:
                    with open(image, 'rb') as file:
                        image_bytes = file.read()
                    input_data.append(image_bytes)
                except Exception as e:
                    return f"Error reading image file: {str(e)}"
            else:
                return "Error: Invalid image file path."

            caption = self.llm_engine(input_data)
            return caption
        except Exception as e:
            return f"Error generating caption: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['require_llm_engine'] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/image_captioner
    python tool.py
    """

    import json

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Image_Captioner_Tool
    # tool = Image_Captioner_Tool()
    tool = Image_Captioner_Tool(model_string="gpt-4o")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/baseball.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Execute the tool with default prompt
    try:
        execution = tool.execute(image=image_path)
        print("Generated Caption:")
        print(json.dumps(execution, indent=4)) 
    except Exception as e: 
        print(f"Execution failed: {e}")

    print("Done!")
