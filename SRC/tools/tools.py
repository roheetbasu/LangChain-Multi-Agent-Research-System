from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """ Search web for recent and reliable information on a topic. Returns Titles, URLS and content"""
    results = tavily.search(query=query, max_results=5)
    
    output = []
    
    for r in results['results']:
        output.append(
            f"Title:{r['title']}\nURL:{r['url']}\nSnippet:{r['content'][:300]}\n"
        )
    return "\n-----\n".join(output)

@tool
def scrape_url(url: str) -> str:
    """Scrape and extract the main readable text content from a webpage URL."""
    
    try:
        # Download the webpage
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )
        response.raise_for_status()

        # Extract the main article/content
        text = trafilatura.extract(
            response.text,
            include_links=True,
            include_tables=True
        )

        # Fallback to BeautifulSoup if extraction fails
        if not text:
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unnecessary elements
            for element in soup([
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]):
                element.decompose()

            text = soup.get_text(
                separator=" ",
                strip=True
            )

        if not text:
            return f"Could not extract readable content from {url}"

        # Clean excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Limit size so we don't send huge webpages to the LLM
        max_chars = 12000
        text = text[:max_chars]

        return f"URL: {url}\n\nContent:\n{text}"

    except requests.exceptions.RequestException as e:
        return f"Failed to fetch URL: {url}\nError: {str(e)}"

    except Exception as e:
        return f"Failed to scrape URL: {url}\nError: {str(e)}"