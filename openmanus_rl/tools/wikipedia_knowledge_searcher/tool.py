import os
import wikipedia

from openmanus_rl.tools.base import BaseTool

class Wikipedia_Knowledge_Searcher_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Wikipedia_Knowledge_Searcher_Tool",
            tool_description="A tool that searches Wikipedia and returns web text based on a given query.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for Wikipedia.",            },
            output_type="dict - A dictionary containing the search results, extracted text, and any error messages.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Python programming language")',
                    "description": "Search Wikipedia for information about Python programming language."
                },
                {
                    "command": 'execution = tool.execute(query="Artificial Intelligence")',
                    "description": "Search Wikipedia for information about Artificial Intelligence"
                },
                {
                    "command": 'execution = tool.execute(query="Theory of Relativity")',
                    "description": "Search Wikipedia for the full article about the Theory of Relativity."
                },
            ],
        )

    def search_wikipedia(self, query, max_length=2000):
        """
        Searches Wikipedia based on the given query and returns the text.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): The maximum length of the returned text. Use -1 for full text.

        Returns:
            tuple: (search_results, page_text)
        """
        try:
            search_results = wikipedia.search(query)
            if not search_results:
                return [], "No results found for the given query."

            page = wikipedia.page(search_results[0])
            text = page.content

            if max_length != -1:
                text = text[:max_length]

            return search_results, text
        except wikipedia.exceptions.DisambiguationError as e:
            return e.options, f"DisambiguationError: {str(e)}"
        except wikipedia.exceptions.PageError:
            return [], f"PageError: No Wikipedia page found for '{query}'."
        except Exception as e:
            return [], f"Error searching Wikipedia: {str(e)}"

    def execute(self, query, max_length=2000):
        """
        Searches Wikipedia based on the provided query and returns the results.

        Parameters:
            query (str): The search query for Wikipedia.
            max_length (int): The maximum length of the returned text. Use -1 for full text.

        Returns:
            dict: A dictionary containing the search results, extracted text, and formatted output.
        """
        search_results, text = self.search_wikipedia(query, max_length)
        
        formatted_output = f"Search results for '{query}':\n"
        formatted_output += "\n".join(f"{i}. {result}" for i, result in enumerate(search_results, 1))
        formatted_output += f"\n\nExtracted text:\n{text}"

        return {
            # "search_results": search_results,
            # "extracted_text": text,
            "output": formatted_output
        }

    def get_metadata(self):
        """
        Returns the metadata for the Wikipedia_Knowledge_Searcher_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd octotools/tools/wikipedia_knowledge_searcher
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Wikipedia_Knowledge_Searcher_Tool
    tool = Wikipedia_Knowledge_Searcher_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Sample query for searching Wikipedia
    # query = "Python programming language"
    query = "kidney"

    import json

    # Execute the tool with the sample query
    try:
        execution = tool.execute(query=query)
        print("Execution Result:")
        print(json.dumps(execution, indent=4))
        for key, value in execution.items():
            print(f"{key}:\n{value}\n")
    except ValueError as e:
        print(f"Execution failed: {e}")

    print("Done!")
