from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True) 
weather_api_key = os.getenv('WEATHER_API_KEY')

mcp = FastMCP("WeatherAPI")

@mcp.tool()
def get_weather_info(location: str) -> str:
    """
    Gets the temperature for a location.
    Args:
        location: Location for which temperature needs to be obtained.
    """
    try:
        url = f"https://api.weatherapi.com/v1/current.json"
        params = {"key": weather_api_key, "q": location}
        response = requests.get(url, params=params, timeout=10)        
        response.raise_for_status()
        data = response.json()
        #Ensure output is a string, not dict
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching ipinfo: {e}"

if __name__ == "__main__":
    mcp.run()
