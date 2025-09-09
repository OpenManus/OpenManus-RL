import os
import requests
from bs4 import BeautifulSoup
import time

from openmanus_rl.tools.base import BaseTool

class Nature_News_Fetcher_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="Nature_News_Fetcher_Tool",
            tool_description="A tool that fetches the latest news articles from Nature.",
            tool_version="1.0.0",
            input_types={
                "num_articles": "int - The number of articles to fetch (default: 100).",
                "max_pages": "int - The maximum number of pages to fetch (default: 5).",
            },
            output_type="list - A list of dictionaries containing information about the latest Nature news articles.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute()',
                    "description": "Fetch the latest 100 news articles from Nature."
                },
                {
                    "command": 'execution = tool.execute(num_articles=50, max_pages=3)',
                    "description": "Fetch the latest 50 news articles from Nature, searching up to 3 pages."
                },
            ],
        )
        self.base_url = "https://www.nature.com/nature/articles"

    def fetch_page(self, page_number):
        """
        Fetches a single page of news articles from Nature's website.

        Parameters:
            page_number (int): The page number to fetch.

        Returns:
            str: The HTML content of the page.
        """
        params = {
            "searchType": "journalSearch",
            "sort": "PubDate",
            "type": "news",
            "page": str(page_number)
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(self.base_url, params=params, headers=headers)
        response.raise_for_status()
        return response.text

    def parse_articles(self, html_content):
        """
        Parses the HTML content and extracts article information.

        Parameters:
            html_content (str): The HTML content of the page.

        Returns:
            list: A list of dictionaries containing article information.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        articles_section = soup.find('section', id='new-article-list')
        if not articles_section:
            return []

        articles = []
        for article in articles_section.find_all('article', class_='c-card'):
            title_elem = article.find('h3', class_='c-card__title')
            title = title_elem.text.strip() if title_elem else "No title found"
            
            url_elem = title_elem.find('a') if title_elem else None
            url = "https://www.nature.com" + url_elem['href'] if url_elem and 'href' in url_elem.attrs else "No URL found"
            
            description_elem = article.find('div', {'data-test': 'article-description'})
            description = description_elem.text.strip() if description_elem else "No description available"
            
            authors_elem = article.find('ul', {'data-test': 'author-list'})
            authors = [author.text.strip() for author in authors_elem.find_all('li')] if authors_elem else ["No authors found"]
            
            date_elem = article.find('time')
            date = date_elem['datetime'] if date_elem and 'datetime' in date_elem.attrs else "No date found"

            image_elem = article.find('img')
            image_url = image_elem['src'] if image_elem and 'src' in image_elem.attrs else "No image found"

            articles.append({
                'title': title,
                'url': url,
                'description': description,
                'authors': authors,
                'date': date,
                'image_url': image_url
            })

        return articles

    def execute(self, num_articles=100, max_pages=5):
        """
        Fetches the latest news articles from Nature's website.

        Parameters:
            num_articles (int): The number of articles to fetch.
            max_pages (int): The maximum number of pages to fetch.

        Returns:
            list: A list of dictionaries containing article information.
        """
        all_articles = []
        page_number = 1

        try:
            while len(all_articles) < num_articles and page_number <= max_pages:
                html_content = self.fetch_page(page_number)
                page_articles = self.parse_articles(html_content)
                
                if not page_articles:
                    break  # No more articles found

                all_articles.extend(page_articles)
                page_number += 1
                time.sleep(1)  # Be polite to the server

            return all_articles[:num_articles]
        except Exception as e:
            return [{"error": str(e)}]

    def get_metadata(self):
        """
        Returns the metadata for the Nature_News_Fetcher_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd octotools/tools/nature_news_fetcher
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Nature_News_Fetcher_Tool
    tool = Nature_News_Fetcher_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    import json
  

    # Execute the tool to fetch the latest 10 articles (for demonstration purposes)
    try:
        execution = tool.execute(num_articles=10, max_pages=1)
        print(json.dumps(execution, indent=4))
        print("\nExecution Result:")
        print(f"Number of articles fetched: {len(execution)}")
        print("\nSample articles:")
        for i, article in enumerate(execution[:10], 1):
            print(f"\n{i}. Title: {article['title']}")
            print(f"   URL: {article['url']}")
            print(f"   Description: {article['description'][:100]}...")  # Show first 100 characters
            print(f"   Authors: {', '.join(article['authors'])}")
            print(f"   Date: {article['date']}")
            print(f"   Image URL: {article['image_url']}")
    except Exception as e:
        print(f"Execution failed: {e}")

    print("Done!")
