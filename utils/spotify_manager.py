import os
import json
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth

class Spotify_Manager:
    def __init__(self):
        self.username = '31fa2cf4yr36midooubemju5tvky'
        self.CLIENT_ID = 'f9f3fd73db6a433299753b1be0dbda04'
        self.CLIENT_SECRET = '91b62e334e1b4a95ae9515fab71fcc2a'
        self.REDIRECT_URI = 'http://google.com/callback/'
        self.SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"

        # File to store user's token
        self.TOKEN_FILE = "../variables/spotify_token.json"

        self.last_url = ""

        self.spotify_client = self.authenticate_user()

    def authenticate_user(self):
        # Check if the token file exists
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, "r") as file:
                token_info = json.load(file)

                return spotipy.Spotify(auth=token_info["access_token"])

        # Authenticate with SpotifyOAuth and save the token
        sp_oauth = SpotifyOAuth(
            client_id=self.CLIENT_ID,
            client_secret=self.CLIENT_SECRET,
            redirect_uri=self.REDIRECT_URI,
            scope=self.SCOPE
        )
        token_info = sp_oauth.get_access_token()
        with open(self.TOKEN_FILE, "w") as file:
            json.dump(token_info, file)
        return spotipy.Spotify(auth=token_info["access_token"])

    def play_playlist_or_album(self, url):
        self.last_url = url
        try:
            # Determine if the URL is a playlist or album
            if "playlist" in url:
                context_type = "playlist"
            elif "album" in url:
                context_type = "album"
            elif "track" in url:
                context_type = "track"
            else:
                print("Invalid URL. Only playlists or albums are supported.")
                return

            # Extract ID from the URL
            context_id = url.split("/")[-1].split("?")[0]

            # Get the user's active device
            devices = self.spotify_client.devices()
            if not devices["devices"]:
                print("No active devices found. Please start playing Spotify on a device first.")
                return
            device_id = devices["devices"][0]["id"]

            if context_type in ["playlist", "album"]:
                # Play a playlist or album
                self.spotify_client.start_playback(device_id=device_id, context_uri=f"spotify:{context_type}:{context_id}")
                print(f"Started playing {context_type}: {url}")
            elif context_type == "track":
                # Play a single track
                self.spotify_client.start_playback(device_id=device_id, uris=[f"spotify:{context_type}:{context_id}"])
                print(f"Started playing track: {url}")

        except spotipy.exceptions.SpotifyException as e:
            print(f"Spotify API error: {e}")
            if "The access token expired" in str(e):
                print("Token expired. Please re-authenticate.")
                os.remove(self.TOKEN_FILE)
                time.sleep(0.5)
                # Retry authentication
                self.authenticate_user()
                # Retry playing the playlist or album
                self.play_playlist_or_album(self.last_url)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    spotify = Spotify_Manager()
    spotify_client = spotify.authenticate_user()
    url = input("Enter Spotify playlist or album URL: ")
    spotify.play_playlist_or_album(url)
