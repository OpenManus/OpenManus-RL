import logging
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from openmanus_rl.tools.base import BaseTool

# Load environment variables from .env if present at process start.
load_dotenv()


class Google_Search_Tool(BaseTool):
  """Google Programmable Search Engine (Custom Search JSON API) tool.

  Reads `GOOGLE_API_KEY` and `GOOGLE_CX` from environment and performs a
  search via `https://www.googleapis.com/customsearch/v1`.
  """

  def __init__(self) -> None:
    super().__init__(
        tool_name="Google_Search_Tool",
        tool_description=(
            "A tool that performs Google searches based on a given text query."),
        tool_version="1.0.1",
        input_types={
            "query": "str - The search query to be used for the Google search.",
            "num_results": "int - Number of results to return (default: 10).",
        },
        output_type=
        "list - A list of dictionaries containing search result information.",
        demo_commands=[
            {
                "command": 'execution = tool.execute(query="Python programming")',
                "description": (
                    "Perform a Google search for 'Python programming' and return the "
                    "default number of results.")
            },
            {
                "command": 'execution = tool.execute(query="Machine learning tutorials", num_results=5)',
                "description": (
                    "Perform a Google search for 'Machine learning tutorials' and "
                    "return 5 results.")
            },
        ],
    )

    # References:
    # - API intro: https://developers.google.com/custom-search/v1/introduction
    # - Engine setup: https://programmablesearchengine.google.com/controlpanel/all
    self.api_key = os.getenv("GOOGLE_API_KEY")
    self.cx = os.getenv("GOOGLE_CX")
    self.base_url = "https://www.googleapis.com/customsearch/v1"

  def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
    """Perform a Google search using Custom Search JSON API.

    Args:
      query: Search query text.
      num_results: Number of results to request (max 10 per page by API).

    Returns:
      Parsed JSON response from the API as a dictionary.
    """
    params = {
        "q": query,
        "key": self.api_key,
        "cx": self.cx,
        "num": num_results,
    }

    try:
      resp = requests.get(self.base_url, params=params, timeout=30)
    except Exception as exc:  # Network or requests errors.
      return {
          "error": {
              "message": f"Request failed: {exc}",
              "status_code": None,
          }
      }

    # Try to decode JSON regardless of status; Google returns structured error.
    try:
      data = resp.json()
    except ValueError:
      return {
          "error": {
              "message": f"Non-JSON response (status {resp.status_code})",
              "status_code": resp.status_code,
              "text": resp.text[:500],
          }
      }

    if resp.status_code != 200:
      # Surface API error clearly to the caller.
      return {
          "error": data.get("error", {
              "message": f"HTTP {resp.status_code}",
              "status_code": resp.status_code,
          })
      }

    return data

  def execute(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """Execute a search and return simplified item list or error.

    Args:
      query: Search query text.
      num_results: Number of results to request.

    Returns:
      A list of result dicts with keys: title, link, snippet; or a single
      item list containing an "error" key with a message.
    """
    if not self.api_key and not self.cx:
      return [{
          "error": (
              "GOOGLE_API_KEY and GOOGLE_CX are not set. Please set both "
              "environment variables.")
      }]
    if not self.api_key:
      return [{
          "error": (
              "Google API key is not set. Please set GOOGLE_API_KEY.")
      }]
    if not self.cx:
      return [{
          "error": (
              "Google Custom Search Engine ID is not set. Please set GOOGLE_CX.")
      }]

    results = self.google_search(query, num_results)
    logging.debug("Google search raw response: %s", results)

    if isinstance(results, dict) and "error" in results:
      err = results["error"]
      # Normalize to message string when possible.
      if isinstance(err, dict) and "message" in err:
        return [{"error": err["message"]}]
      return [{"error": str(err)}]

    try:
      items = results.get("items", []) if isinstance(results, dict) else []
      if not items:
        return [{"error": "No results found."}]
      return [{
          "title": item.get("title", ""),
          "link": item.get("link", ""),
          "snippet": item.get("snippet", ""),
      } for item in items]
    except Exception as exc:
      return [{"error": f"Failed to parse results: {exc}"}]

  def get_metadata(self):
    """Return the metadata for the Google_Search_Tool."""
    metadata = super().get_metadata()
    return metadata


if __name__ == "__main__":
  # Minimal self-test for manual runs.
  tool = Google_Search_Tool()
  out = tool.execute(query="nobel prize winners in chemistry 2024", num_results=5)
  print(out)
