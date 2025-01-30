import os
import json
import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth

class Spotify_Manager:
    def __init__(self):

        self.username, self.CLIENT_ID, self.CLIENT_SECRET = self.load_credentials()

        self.REDIRECT_URI = 'http://google.com/callback/'
        self.SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"

        # File to store user's token
        self.TOKEN_FILE = "variables/spotify_token.json"
        self.CONFIG_FILE = "variables/spotify_config.json"

        self.last_url = ""
        self.currently_playing_url = ""

        self.is_authenticated = False
        self.spotify_client = self.authenticate_user()
        self.is_authenticated = True

        #self.currently_playing = self.spotify_client.current_playback()

    def load_credentials(self):
        """Load credentials from a JSON file and set them as global variables."""
        if not os.path.exists(self.CONFIG_FILE):
            raise FileNotFoundError("Config file not found. Run save_credentials() first.")

        with open(self.CONFIG_FILE, "r") as file:
            credentials = json.load(file)

        return credentials["username"], credentials["CLIENT_ID"], credentials["CLIENT_SECRET"]

    def authenticate_user(self):
        print("Authenticating user...")
        # Check if the token file exists
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, "r") as file:
                print("Token file found.")
                token_info = json.load(file)

                print("Is token expired:", spotipy.SpotifyOAuth.is_token_expired(token_info))

                if not spotipy.SpotifyOAuth.is_token_expired(token_info):
                    return spotipy.Spotify(auth=token_info["access_token"])
                else:
                    print(spotipy.SpotifyOAuth.refresh_access_token(token_info["refresh_token"]))

            #print("Token expired. Please re-authenticate.")
            #os.remove(self.TOKEN_FILE)
            #time.sleep(0.1)

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
        print("User authenticated.")
        return spotipy.Spotify(auth=token_info["access_token"])

    def play_playlist_or_album(self, url):
        self.last_url = url
        self.currently_playing_url = url

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

    def find_first_non_empty(self, queue):
        local_queue = queue

        for x in local_queue:
            if x != "":
                print(f"Playing {x}")
                self.play_playlist_or_album(x)
            else:
                print("Empty string in queue, skipping")
                local_queue.remove(x)

        return local_queue

    def continue_queue(self, queue):
        if not self.is_authenticated:
            print("Not authenticated")
            return queue

        if self.currently_playing_url == "":
            print("No current song, playlist ecc. Playing")
            print("Skipping...")
            local_queue = self.find_first_non_empty(queue)

        elif self.spotify_client.current_playback() is None:
            print("No current song, playlist ecc. Playing")
            print("Skipping...")
            local_queue = self.find_first_non_empty(queue)

        elif not (str(self.spotify_client.current_playback().get('context', {}).get('external_urls', {}).get('spotify', None)) in self.currently_playing_url):
            print("Current song is not the same as the one in the queue. Playing")
            print("Skipping...")
            local_queue = self.find_first_non_empty(queue)

        else:
            local_queue = queue
            print("Nothing to do, waiting for the song to finish")

        print("NOW PLAYING URL:", self.currently_playing_url)
        print("NOW PLAYING:", self.spotify_client.current_playback().get('context', {}).get('external_urls', {}).get('spotify', None))

        return local_queue

if __name__ == "__main__":
    spotify = Spotify_Manager()
    url = input("Enter Spotify playlist or album URL: ")
    spotify.play_playlist_or_album(url)
