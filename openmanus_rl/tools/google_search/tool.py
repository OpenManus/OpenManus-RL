import os
import requests
from typing import List, Dict, Any

from openmanus_rl.tools.base import BaseTool

from dotenv import load_dotenv
load_dotenv()

class Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Google_Search_Tool",
            tool_description="A tool that performs Google searches based on a given text query.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to be used for the Google search.",
                "num_results": "int - The number of search results to return (default: 10).",
            },
            output_type="list - A list of dictionaries containing search result information.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Python programming")',
                    "description": "Perform a Google search for 'Python programming' and return the default number of results."
                },
                {
                    "command": 'execution = tool.execute(query="Machine learning tutorials", num_results=5)',
                    "description": "Perform a Google search for 'Machine learning tutorials' and return 5 results."
                },
            ],
        )
        # self.api_key = os.getenv("GOOGLE_API_KEY")
        self.api_key = os.getenv("GOOGLE_API_KEY") # NOTE: Replace with your own API key (Ref: https://developers.google.com/custom-search/v1/introduction)
        self.cx = os.getenv("GOOGLE_CX") # NOTE: Replace with your own custom search (Ref: https://programmablesearchengine.google.com/controlpanel/all)
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Performs a Google search using the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return.

        Returns:
            Dict[str, Any]: The raw search results from the Google API.
        """
        params = {
            'q': query,
            'key': self.api_key,
            'cx': self.cx,
            'num': num_results
        }
        
        response = requests.get(self.base_url, params=params)
        return response.json()

    def execute(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Executes a Google search based on the provided query.

        Parameters:
            query (str): The search query.
            num_results (int): The number of search results to return (default: 10).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search result information.
        """
        if not self.api_key:
            return [{"error": "Google API key is not set. Please set the GOOGLE_API_KEY environment variable."}]

        try:
            results = self.google_search(query, num_results)
            print(results)
            
            if 'items' in results:
                return [
                    {
                        "title": item['title'],
                        "link": item['link'],
                        "snippet": item['snippet']
                    }
                    for item in results['items']
                ]
            else:
                return [{"error": "No results found."}]
        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]

    def get_metadata(self):
        """
        Returns the metadata for the Google_Search_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    export GOOGLE_API_KEY=your_api_key_here
    cd octotools/tools/google_search
    python tool.py
    """

    # Example usage of the Google_Search_Tool
    tool = Google_Search_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Execute the tool to perform a Google search
    query = "nobel prize winners in chemistry 2024"
    try:
        execution = tool.execute(query=query, num_results=5)
        print("\nExecution Result:")
        print(f"Search query: {query}")
        print(f"Number of results: {len(execution)}")
        print("\nSearch Results:")
        if "error" in execution[0]:
            print(f"Error: {execution[0]['error']}")
        else:
            for i, item in enumerate(execution, 1):
                print(f"\n{i}. Title: {item['title']}")
                print(f"   URL: {item['link']}")
                print(f"   Snippet: {item['snippet']}")
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
