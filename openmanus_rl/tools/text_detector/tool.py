# octotools/tools/text_detector/tool.py

import os
import time
from openmanus_rl.tools.base import BaseTool

import warnings
warnings.filterwarnings("ignore")

class Text_Detector_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Text_Detector_Tool",
            tool_description="A tool that detects text in an image using EasyOCR.",
            tool_version="1.0.0",
            input_types={
                "image": "str - The path to the image file.",
                "languages": "list - A list of language codes for the OCR model.",
                "detail": "int - The level of detail in the output. Set to 0 for simpler output, 1 for detailed output."
            },
            output_type="list - A list of detected text blocks.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", languages=["en"])',
                    "description": "Detect text in an image using the default language (English)."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", languages=["en", "de"])',
                    "description": "Detect text in an image using multiple languages (English and German)."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/image.png", languages=["en"], detail=0)',
                    "description": "Detect text in an image with simpler output (text without coordinates and scores)."
                },
            ],
            user_metadata={
                "frequently_used_language": {
                    "ch_sim": "Simplified Chinese",
                    "ch_tra": "Traditional Chinese",
                    "de": "German",
                    "en": "English",
                    "es": "Spanish",
                    "fr": "French",
                    "hi": "Hindi",
                    "ja": "Japanese",
                }
            }
        )

    def build_tool(self, languages=None):
        """
        Builds and returns the EasyOCR reader model.

        Parameters:
            languages (list): A list of language codes for the OCR model.

        Returns:
            easyocr.Reader: An initialized EasyOCR Reader object.
        """
        languages = languages or ["en"]  # Default to English if no languages provided
        try:
            import easyocr
            reader = easyocr.Reader(languages)
            return reader
        except ImportError:
            raise ImportError("Please install the EasyOCR package using 'pip install easyocr'.")
        except Exception as e:
            print(f"Error building the OCR tool: {e}")
            return None
    
    def execute(self, image, languages=None, max_retries=10, retry_delay=5, clear_cuda_cache=False, **kwargs):
        """
        Executes the OCR tool to detect text in the provided image.

        Parameters:
            image (str): The path to the image file.
            languages (list): A list of language codes for the OCR model.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int): Delay in seconds between retry attempts.
            clear_cuda_cache (bool): Whether to clear CUDA cache on out-of-memory errors.
            **kwargs: Additional keyword arguments for the OCR reader.

        Returns:
            list: A list of detected text blocks.
        """
        languages = languages or ["en"]

        for attempt in range(max_retries):
            try:
                reader = self.build_tool(languages)
                if reader is None:
                    raise ValueError("Failed to build the OCR tool.")
                
                result = reader.readtext(image, **kwargs)
                try:
                    # detail = 1: Convert numpy types to standard Python types
                    cleaned_result = [
                        ([[int(coord[0]), int(coord[1])] for coord in item[0]], item[1], round(float(item[2]), 2))
                        for item in result
                    ]
                    return cleaned_result
                except Exception as e:
                    # detail = 0
                    return result

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
                print(f"Error detecting text: {e}")
                break
        
        print(f"Failed to detect text after {max_retries} attempts.")
        return []

    def get_metadata(self):
        """
        Returns the metadata for the Text_Detector_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/text_detector
    python tool.py
    """
    import json

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Text_Detector_Tool
    tool = Text_Detector_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    # relative_image_path = "examples/chinese_tra.jpg"
    # relative_image_path = "examples/chinese.jpg"
    relative_image_path = "examples/english.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Execute the tool
    try:
        # execution = tool.execute(image=image_path, languages=["en", "ch_sim"])
        # execution = tool.execute(image=image_path, languages=["en", "ch_tra"])
        execution = tool.execute(image=image_path, languages=["en"])
        print(json.dumps(execution))

        print("Detected Text:", execution)
    except ValueError as e:
        print(f"Execution failed: {e}")

    print("Done!")
