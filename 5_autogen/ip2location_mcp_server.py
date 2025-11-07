from mcp.server.fastmcp import FastMCP
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True) 
ip2location_api_key = os.getenv('IP2LOCATION_API_KEY')

mcp = FastMCP("Ipinfo")

@mcp.tool()
def get_ip_info(ip: str) -> str:
    """
    Gets the location and other details for an ip address.
    Args:
        ip: IPv4 address for which details need to be obtained.
    """
    try:
        url = f"https://api.ip2location.io/?key={ip2location_api_key}&ip={ip}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        #Ensure output is a string, not dict
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching ipinfo: {e}"

if __name__ == "__main__":
    mcp.run()
