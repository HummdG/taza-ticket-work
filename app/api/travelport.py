"""
Travelport API authentication and header management
"""

import os
import requests
from typing import Dict


# Travelport API Configuration
CLIENT_ID = os.getenv("TRAVELPORT_APPLICATION_KEY")
CLIENT_SECRET = os.getenv("TRAVELPORT_APPLICATION_SECRET")
USERNAME = os.getenv("TRAVELPORT_USERNAME")
PASSWORD = os.getenv("TRAVELPORT_PASSWORD")
ACCESS_GROUP = os.getenv("TRAVELPORT_ACCESS_GROUP")

OAUTH_URL = "https://oauth.pp.travelport.com/oauth/oauth20/token"
CATALOG_URL = "https://api.pp.travelport.com/11/air/catalog/search/catalogproductofferings"


def fetch_password_token() -> str:
    """
    Get OAuth token from Travelport API
    
    Returns:
        str: Access token for API authentication
        
    Raises:
        requests.HTTPError: If authentication fails
    """
    data = {
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "openid"
    }
    
    response = requests.post(
        OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=data
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_api_headers() -> Dict[str, str]:
    """
    Build headers for Travelport API requests
    
    Returns:
        Dict[str, str]: Complete headers dictionary for API requests
        
    Raises:
        requests.HTTPError: If token generation fails
        ValueError: If required environment variables are missing
    """
    if not ACCESS_GROUP:
        raise ValueError("TRAVELPORT_ACCESS_GROUP environment variable is required")
    
    token = fetch_password_token()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Authorization": f"Bearer {token}",
        "XAUTH_TRAVELPORT_ACCESSGROUP": ACCESS_GROUP,
        "Accept-Version": "11",
        "Content-Version": "11",
    } 