from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
import os
import json
import datetime

load_dotenv(override=True) 
news_api_key = os.getenv('NEWS_API_KEY')

mcp = FastMCP("NewsAPI")
yesterday = datetime.date.today() - datetime.timedelta(days=1)

@mcp.tool()
def get_search_results(searchFor: str) -> str:
    """
    Obtains news about the specified topic in searchFor parameter.
    Args:
        searchFor: Our query for which we need news results.
    """
    try:
        url = f"https://newsapi.org/v2/everything?q={searchFor}&from={yesterday}&sortBy=popularity&apiKey={news_api_key}"        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        #Ensure output is a string, not dict
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching ipinfo: {e}"

if __name__ == "__main__":
    mcp.run()
