import os
import json
import random
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class Song:
    def __init__(self, url, duration, title, artist, album, cover_url, progress, original_url):
        self.url = url
        self.duration = duration
        self.title = title
        self.artist = artist
        self.album = album
        self.cover_url = cover_url
        self.progress = progress  # Current playback position
        self.original_url = original_url  # Original URL used to play the song

    def to_dict(self):
        """Convert song info to a dictionary for saving."""
        return {
            "url": self.url,
            "duration": self.duration,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "cover_url": self.cover_url,
            "original_url": self.original_url
        }


class Spotify_Manager:
    def __init__(self):

        # File to store user's token
        self.TOKEN_FILE = "variables/spotify_token.json"
        self.CONFIG_FILE = "variables/spotify_config.json"
        self.username, self.CLIENT_ID, self.CLIENT_SECRET = self.load_credentials()
        self.REDIRECT_URI = 'http://google.com/callback/'
        self.SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"


        # Queue of songs to play
        self.queue = []
        self.active_sources = []
        self.current_song = None

        # URLs
        self.last_url = ""
        self.currently_playing_url = ""
        self.time_to_wait = 0

        self.already_played_tracks = []

        self.is_authenticated = False
        self.spotify_client = self.authenticate_user()
        self.is_authenticated = True

        #self.currently_playing = self.spotify_client.current_playback()

    """SETTERS AND GETTERS"""
    def remove_item_from_active_sources(self, item):
        try:
            self.active_sources.remove(item)
        except ValueError:
            print("Item not found in active sources")

        # check if item is running:
        if self.current_song.original_url == item:
            self.spotify_client.pause_playback()
            self.play_next_song()
            #self.current_song = None
            #self.time_to_wait = 0

        self.active_sources.remove(item)

    def add_item_to_active_sources(self, item):
        self.active_sources.append(item)

    """AUTHENTICATION"""
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

    """PLAYLIST/ALBUM PLAYBACK"""
    def get_source_type(self, url):
        if "playlist" in url:
            return "playlist"
        elif "album" in url:
            return "album"
        elif "track" in url:
            return "track"
        else:
            return None

    def get_current_song_info(self, url):
        """Retrieve current playing song details and store them."""
        try:
            track_id = url.split("/")[-1].split("?")[0]
            track = self.spotify_client.track(track_id)  # Fetch track details

            song_info = Song(
                url=track['external_urls']['spotify'],
                duration=track['duration_ms'],
                title=track['name'],
                artist=", ".join([artist['name'] for artist in track['artists']]),
                album=track['album']['name'],
                cover_url=track['album']['images'][0]['url'],  # High-res album cover
                original_url=url
            )

            self.current_song = song_info
            return song_info

        except spotipy.exceptions.SpotifyException as e:
            print(f"Spotify API error: {e}")
            return None

    def play_playlist_or_album(self, url):
        # legacy can be removed after some tweaking
        self.last_url = url
        self.currently_playing_url = url

        song_info = self.get_current_song_info(url)

        try:
            # Determine if the URL is a playlist or album
            context_type = self.get_source_type(url)

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

            print(f"Started playing {context_type}: {url}")
            return song_info

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

    def get_random_not_already_played_song_in_playlist(self, source):
        # check if source is an album, playlist or song

        type = self.get_source_type(source)

        if type == "track":
            return source
        if type == "playlist":
            playlist_album = self.spotify_client.playlist_items(source)
        elif type == "album":
            playlist_album = self.spotify_client.album_tracks(source)
        else:
            raise ValueError("Invalid source type")

        tracks = playlist_album['items']
        random.shuffle(tracks)

        while True:
            track = random.choice(tracks)
            if track['track']['external_urls']['spotify'] not in self.already_played_tracks:
                print("Found a new song to play")
                return track['track']['external_urls']['spotify']
            else:
                print("Already played this song, skipping...")
                tracks.remove(track)

    """QUEUE MANAGEMENT"""
    def play_next_song(self):
        # RESET JUST TO BE SURE
        self.current_song = None
        self.time_to_wait = 0

        source = random.choice(self.active_sources)
        track = self.get_random_not_already_played_song_in_playlist(source)
        return self.play_playlist_or_album(track)

    def handle_music(self):
        while True:
            if not self.is_authenticated:
                print("Not authenticated")
                time.sleep(2)

            if len(self.active_sources) > 0:
                if self.current_song is None:
                    print("No current song, playing next one...")
                    info = self.play_next_song()
                    self.time_to_wait = info.duration

                # check this only every 30 seconds or exactly when the song ends
                elif self.time_to_wait == 0:
                    if self.spotify_client.current_playback() is None:
                        print("No current song, playing next one...")
                        info = self.play_next_song()
                        self.time_to_wait = info.duration

            else:
                print("No active sources, waiting...")


            time.sleep(1)

        # When playing a song save how long it is and 5 seconds before it ends start checking if the song is still playing
        # If it is not playing, skip to a next random song from the sources !not the playlist, but ensure that a song is not played twice if there is another option!
        # If the song is still playing, continue checking until it ends



if __name__ == "__main__":
    spotify = Spotify_Manager()
    url = input("Enter Spotify playlist or album URL: ")
    spotify.play_playlist_or_album(url)

    # elif not (str(self.spotify_client.current_playback().get('context', {}).get('external_urls', {}).get('spotify', None)) in self.currently_playing_url):

