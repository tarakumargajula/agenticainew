from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

mcp = FastMCP("GutenbergAPI")

@mcp.tool()
def get_book_details(author: str, title: str) -> dict:
    """Search for a book by author and title in Project Gutenberg."""
    try:
        url = f"https://gutendex.com/books/?search={author}%20{title}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "count": data.get("count", 0),
            "results": [
                {
                    "title": b.get("title"),
                    "authors": [a.get("name") for a in b.get("authors", [])],
                }
                for b in data.get("results", [])[:5]
            ],
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()
