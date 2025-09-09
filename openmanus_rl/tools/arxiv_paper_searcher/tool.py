import re
import requests
from bs4 import BeautifulSoup

from openmanus_rl.tools.base import BaseTool

class ArXiv_Paper_Searcher_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="ArXiv_Paper_Searcher_Tool",
            tool_description="A tool that searches arXiv for papers based on a given query.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for arXiv papers.",
                "size": "int - The number of results per page (25, 50, 100, or 200). If None, use 25.",
                "max_results": "int - The maximum number of papers to return (default: 25). Should be less than or equal to 100."
            },
            output_type="list - A list of dictionaries containing paper information.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="tool agents with large language models")',
                    "description": "Search for papers about tool agents with large language models."
                },
                {
                    "command": 'execution = tool.execute(query="quantum computing", size=100, max_results=50)',
                    "description": "Search for quantum computing papers, with 100 results per page, returning a maximum of 50 papers."
                },
                {
                    "command": 'execution = tool.execute(query="machine learning", max_results=75)',
                    "description": "Search for machine learning papers, returning a maximum of 75 papers."
                },
            ],
            user_metadata={
                "valid_sizes": [25, 50, 100, 200],
                "base_url": "https://arxiv.org/search/"
            }
        )

    def build_tool(self):
        """
        No specific build required for this tool.
        """
        pass

    def execute(self, query, size=None, max_results=25):
        """
        Executes the arXiv search tool to find papers based on the given query.

        Parameters:
            query (str): The search query for arXiv papers.
            size (int): The number of results per page.
            max_results (int): The maximum number of papers to return.

        Returns:
            list: A list of dictionaries containing paper information.
        """
        valid_sizes = self.user_metadata["valid_sizes"]
        base_url = self.user_metadata["base_url"]

        if size is None:
            size = 25
        elif size not in valid_sizes:
            size = min(valid_sizes, key=lambda x: abs(x - size))

        results = []
        start = 0

        max_results = min(max_results, 100) # NOTE: For traffic reasons, limit to 100 results

        while len(results) < max_results:
            params = {
                "searchtype": "all",
                "query": query,
                "abstracts": "show",
                "order": "",
                "size": str(size),
                "start": str(start)
            }

            try:
                response = requests.get(base_url, params=params)
                soup = BeautifulSoup(response.content, 'html.parser')

                papers = soup.find_all("li", class_="arxiv-result")
                if not papers:
                    break

                for paper in papers:
                    if len(results) >= max_results:
                        break

                    title = paper.find("p", class_="title").text.strip()
                    authors = paper.find("p", class_="authors").text.strip()
                    authors = re.sub(r'^Authors:\s*', '', authors)
                    authors = re.sub(r'\s+', ' ', authors).strip()
                    
                    abstract = paper.find("span", class_="abstract-full").text.strip()
                    abstract = abstract.replace("△ Less", "").strip()
                    
                    link = paper.find("p", class_="list-title").find("a")["href"]

                    results.append({
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "link": f"{link}"
                    })

                start += size

            except Exception as e:
                print(f"Error searching arXiv: {e}")
                break

        return results[:max_results]

    def get_metadata(self):
        """
        Returns the metadata for the ArXiv_Paper_Searcher_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/arxiv_paper_searcher
    python tool.py
    """

    import json

    print("ArXiv Search Tool Test")

    # Example usage of the ArXiv_Paper_Searcher_Tool
    tool = ArXiv_Paper_Searcher_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    # Sample query for searching arXiv
    query = "enhance mathematical reasoning with large language models"
    # Execute the tool
    try:
        execution = tool.execute(query=query, size=50, max_results=10)
        print("\n==>> Execution:")
        print(json.dumps(execution, indent=4))  # Pretty print JSON
        print("\n==>> Search Results:")
        for i, paper in enumerate(execution, 1):
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {paper['authors']}")
            print(f"   Abstract: {paper['abstract'][:2000]}")
            print(f"   Link: {paper['link']}")
            print()
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
