import os
import requests
from bs4 import BeautifulSoup

from openmanus_rl.tools.base import BaseTool

class URL_Text_Extractor_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="URL_Text_Extractor_Tool",
            tool_description="A tool that extracts all text from a given URL.",
            tool_version="1.0.0",
            input_types={
                "url": "str - The URL from which to extract text.",
            },
            output_type="dict - A dictionary containing the extracted text and any error messages.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(url="https://example.com")',
                    "description": "Extract all text from the example.com website."
                },
                {
                    "command": 'execution = tool.execute(url="https://en.wikipedia.org/wiki/Python_(programming_language)")',
                    "description": "Extract all text from the Wikipedia page about Python programming language."
                },
            ],
        )

    def extract_text_from_url(self, url):
        """
        Extracts all text from the given URL.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            str: The extracted text.
        """
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text = text[:10000] # Limit the text to 10000 characters
            return text
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def execute(self, url):
        extracted_text = self.extract_text_from_url(url)
        return {
            "url": url,
            "extracted_text": extracted_text
        }
    
    def get_metadata(self):
        """
        Returns the metadata for the URL_Text_Extractor_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd octotools/tools/url_text_extractor
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the URL_Text_Extractor_Tool
    tool = URL_Text_Extractor_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Sample URL for extracting text
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

    import json

    # Execute the tool with the sample URL
    try:
        execution = tool.execute(url=url)
        print("Execution Result:")
        print(json.dumps(execution, indent=4))
        for key, value in execution.items():
            print(f"{key}:\n{value}\n")
    except ValueError as e:
        print(f"Execution failed: {e}")

    print("Done!")
