import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


def get_access_token():
    """Lấy access token từ Spotify bằng client credentials flow"""
    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(" SPOTIFY_CLIENT_ID hoặc SPOTIFY_CLIENT_SECRET chưa được thiết lập")

    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_str}"
    }
    data = {
        "grant_type": "client_credentials"
    }

    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)

    if response.status_code != 200:
        raise Exception(f" Lỗi lấy access token: {response.status_code} - {response.text}")

    token = response.json().get("access_token")
    return token


def search_track(query, token=None):
    """Tìm bài hát trên Spotify và trả về thông tin mở rộng"""
    if token is None:
        token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }

    response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f" Spotify API lỗi: {response.status_code} - {response.text}")

    results = response.json()
    items = results.get("tracks", {}).get("items", [])

    if not items:
        return None

    track = items[0]
    return {
        "song": track["name"],
        "artist": track["artists"][0]["name"],
        "url": track["external_urls"]["spotify"],
        "image": track["album"]["images"][0]["url"] if track["album"]["images"] else None
    }
