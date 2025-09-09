import os
import json
from pymed import PubMed
from metapub import PubMedFetcher
from openmanus_rl.tools.base import BaseTool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Suppress stderr by redirecting it to /dev/null
import sys
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")


class Pubmed_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Pubmed_Search_Tool",
            tool_description="A tool that searches PubMed Central to retrieve relevant article abstracts based on a given list of text queries. Use this ONLY if you cannot use the other more specific ontology tools.",
            tool_version="1.0.0",
            input_types={
                "queries": "list[str] - list of queries terms for searching PubMed."
            },
            output_type="list - List of items matching the search query. Each item consists of the title, abstract, keywords, and URL of the article. If no results found, a string message is returned.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(queries=["scoliosis", "injury"])',
                    "description": "Search for PubMed articles mentioning 'scoliosis' OR 'injury'."
                },
                {
                    "command": 'execution = tool.execute(queries=["COVID", "vaccine", "occupational health"])',
                    "description": "Search for PubMed articles mentioning 'COVID' OR 'vaccine' OR 'occupational health'."
                }
            ],
            user_metadata={
                'limitations': "Try to use shorter and more general search queries."
            }
        )
        self.pubmed = PubMed(tool="MyTool", email="my@email.address")
        self.fetch = PubMedFetcher()

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def search_query(self, query_str, max_results=10):
        return self.pubmed.query(query_str, max_results=max_results)

    def execute(self, queries, max_results=10):
        try:
            query_str = f"({'[Title/Abstract] OR '.join(queries) + '[Title/Abstract]'}) AND hasabstract[All Fields] AND fha[Filter]"
            max_results = min(max_results, 50)

            results = self.search_query(query_str, max_results=max_results) # API can only get most recent

            items = []
            for article in results:
                try:
                    article = json.loads(article.toJSON())
                    pubmed_id = article['pubmed_id'] # get id using pymed then get content using metapub

                    article = self.fetch.article_by_pmid(pubmed_id)
                    items.append({
                        'title': article.title,
                        'abstract': article.abstract,
                        'keywords': article.keywords,
                        'url': article.url
                    })
                except:
                    continue

            if len(items) == 0:
                return "No results found for search query. Try another query or tool."
            
            return items
        
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd octotools/tools/pubmed_search
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage
    tool = Pubmed_Search_Tool()

    # Queries
    queries = ["COVID occupational health"]

    # Execute the tool
    try:
        execution = tool.execute(queries=queries)
        print(execution)
    except ValueError as e: 
        print(f"Execution failed: {e}")

    print("Done!")
