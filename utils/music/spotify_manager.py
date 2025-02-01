import os
import json
import random
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class Song:
    def __init__(self, url, duration, title, artist, album, cover_url, original_url, source=None):
        self.url = url
        self.duration = duration
        self.title = title
        self.artist = artist
        self.album = album
        self.cover_url = cover_url
        self.original_url = original_url  # The URL used to play the song (e.g. track URL)
        self.source = source              # The active source from which this song was chosen

    def to_dict(self):
        """Convert song info to a dictionary for saving."""
        return {
            "url": self.url,
            "duration": self.duration,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "cover_url": self.cover_url,
            "original_url": self.original_url,
            "source": self.source
        }


class Spotify_Manager:
    def __init__(self):
        self.running = True

        # Files to store token and config
        self.TOKEN_FILE = "variables/spotify_token.json"
        self.CONFIG_FILE = "variables/spotify_config.json"
        self.username, self.CLIENT_ID, self.CLIENT_SECRET = self.load_credentials()
        self.REDIRECT_URI = 'http://google.com/callback/'
        self.SCOPE = "user-read-playback-state user-modify-playback-state playlist-read-private"

        # Data for managing playback
        self.searching = False
        self.already_played_tracks = {}  # keys: active source, values: list of track URLs already played from that source
        self.active_sources = []         # list of active source URLs (or identifiers)
        self.current_song = None         # the Song object currently playing
        self.time_to_wait = 0            # remaining time (in seconds) for the current song

        # URLs (for internal tracking)
        self.last_url = ""
        self.currently_playing_url = ""

        self.is_authenticated = False
        self.spotify_client = self.authenticate_user()
        self.is_authenticated = True

    """SETTERS AND GETTERS"""

    def remove_item_from_active_sources(self, item):
        print("Removing item from active sources in Spotify_Manager")
        try:
            self.active_sources.remove(item)
            if item in self.already_played_tracks:
                del self.already_played_tracks[item]
        except ValueError:
            print("Item not found in active sources")

        # If the current song came from the removed source, handle it now.
        if self.current_song and self.current_song.source == item:
            print("The current song originated from the removed source.")
            if self.active_sources:
                print("Other active sources exist. Transitioning to the next song.")
                self.spotify_client.pause_playback()
                self.play_next_song()
            else:
                print("No active sources remain. Pausing playback and resetting variables.")
                self.spotify_client.pause_playback()
                self.current_song = None
                self.time_to_wait = 0
        else:
            print("Current song does not belong to the removed source. Continuing playback.")

    def add_item_to_active_sources(self, item):
        self.active_sources.append(item)
        print(f"Added new active source: {item}")
        # If no song is currently playing, start playback.
        if self.current_song is None:
            print("No song currently playing. Starting playback.")
            self.play_next_song()

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
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, "r") as file:
                print("Token file found.")
                token_info = json.load(file)
                if not spotipy.SpotifyOAuth.is_token_expired(token_info):
                    return spotipy.Spotify(auth=token_info["access_token"])
                else:
                    sp_oauth = SpotifyOAuth(
                        client_id=self.CLIENT_ID,
                        client_secret=self.CLIENT_SECRET,
                        redirect_uri=self.REDIRECT_URI,
                        scope=self.SCOPE
                    )
                    token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
                    with open(self.TOKEN_FILE, "w") as file:
                        json.dump(token_info, file)
                    return spotipy.Spotify(auth=token_info["access_token"])

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

    def get_current_song_info(self, url, source=None):
        """Retrieve current playing song details and store them.
           The 'source' parameter indicates which active source this song came from.
        """
        try:
            track_id = url.split("/")[-1].split("?")[0]
            track = self.spotify_client.track(track_id)
            song_info = Song(
                url=track['external_urls']['spotify'],
                duration=track['duration_ms'],
                title=track['name'],
                artist=", ".join([artist['name'] for artist in track['artists']]),
                album=track['album']['name'],
                cover_url=track['album']['images'][0]['url'],
                original_url=url,
                source=source
            )
            self.current_song = song_info
            return song_info

        except spotipy.exceptions.SpotifyException as e:
            print(f"Spotify API error: {e}")
            return None

    def play_playlist_or_album(self, url, source=None):
        """
        Play a given URL which may be a playlist, album, or track.
        The optional 'source' parameter is used to mark which active source this song came from.
        """
        self.last_url = url
        self.currently_playing_url = url

        song_info = self.get_current_song_info(url, source=source)

        try:
            context_type = self.get_source_type(url)
            context_id = url.split("/")[-1].split("?")[0]
            devices = self.spotify_client.devices()
            if not devices["devices"]:
                print("No active devices found. Please start Spotify on a device first.")
                return
            # Using the first available device (adjust the index as needed)
            device_id = devices["devices"][0]["id"]

            if context_type in ["playlist", "album"]:
                self.spotify_client.start_playback(
                    device_id=device_id,
                    context_uri=f"spotify:{context_type}:{context_id}"
                )
                print(f"Started playing {context_type}: {url}")
            elif context_type == "track":
                self.spotify_client.start_playback(
                    device_id=device_id,
                    uris=[f"spotify:{context_type}:{context_id}"]
                )
                print(f"Started playing track: {url}")

            return song_info

        except spotipy.exceptions.SpotifyException as e:
            print(f"Spotify API error: {e}")
            if "The access token expired" in str(e):
                print("Token expired. Re-authenticating...")
                os.remove(self.TOKEN_FILE)
                time.sleep(0.5)
                self.authenticate_user()
                self.play_playlist_or_album(self.last_url, source=source)

        except Exception as e:
            print(f"Error: {e}")

    def get_random_not_already_played_song_in_playlist(self, source):
        source_type = self.get_source_type(source)
        if source_type == "track":
            return source

        if source_type == "playlist":
            playlist_album = self.spotify_client.playlist_items(source)
        elif source_type == "album":
            playlist_album = self.spotify_client.album_tracks(source)
        else:
            raise ValueError("Invalid source type")

        tracks = playlist_album['items']
        random.shuffle(tracks)

        while True:
            if len(tracks) == 0:
                print("No more songs to play, resetting playlist history...")
                self.already_played_tracks[source] = []
                return self.get_random_not_already_played_song_in_playlist(source)
            else:
                track_item = random.choice(tracks)
                played = self.already_played_tracks.get(source, [])
                if source_type == "playlist":
                    track_obj = track_item.get('track')
                    if not track_obj:
                        print("No track data found; skipping this item.")
                        tracks.remove(track_item)
                        continue
                    external_urls = track_obj.get('external_urls', {})
                    track_url = external_urls.get('spotify')
                    if not track_url:
                        print("Spotify URL not found for track; skipping.")
                        tracks.remove(track_item)
                        continue
                elif source_type == "album":
                    external_urls = track_item.get('external_urls', {})
                    track_url = external_urls.get('spotify')
                    if not track_url:
                        print("Spotify URL not found for album track; skipping.")
                        tracks.remove(track_item)
                        continue

                if track_url not in played:
                    print("Found a new song to play.")
                    return track_url
                else:
                    print("Already played this song; skipping.")
                    tracks.remove(track_item)

    """QUEUE MANAGEMENT"""

    def play_next_song(self):
        if len(self.active_sources) == 0:
            print("No active sources to play from.")
            self.spotify_client.pause_playback()
            self.searching = False
            self.current_song = None
            self.time_to_wait = 0
            return None

        self.searching = True
        # Reset current song and timer
        self.current_song = None
        self.time_to_wait = 0

        # Randomly select one of the active sources
        source = random.choice(self.active_sources)
        track = self.get_random_not_already_played_song_in_playlist(source)
        self.already_played_tracks.setdefault(source, []).append(track)

        info = self.play_playlist_or_album(track, source=source)
        if info is not None:
            # Convert milliseconds to seconds
            self.time_to_wait = info.duration // 1000
            print(f"Playing next song: '{info.title}' by {info.artist}. Duration: {self.time_to_wait} seconds")
        else:
            print("Failed to play next song.")
        self.searching = False
        return info

    def handle_music(self):
        """
        Main loop that continuously monitors playback. It ensures that:
          - When a song is nearly finished (0.5 seconds remaining), it starts the next song.
          - If playback stops unexpectedly while active sources exist, it starts the next song.
          - If there are no active sources, it pauses playback.
        This method is intended to be run in parallel with code that adds or removes active sources.
        """
        while self.running:
            if not self.is_authenticated:
                print("Not authenticated. Waiting...")
                time.sleep(2)
                continue

            # If no active sources exist, ensure playback is paused.
            if len(self.active_sources) == 0:
                if self.spotify_client.current_playback().get('is_playing', False):
                    self.spotify_client.pause_playback()
                print("No active sources; waiting...")
                self.current_song = None
                self.time_to_wait = 0
                time.sleep(0.1)
                continue

            # If no song is playing, start the next one.
            if self.current_song is None:
                print("No current song; playing next one...")
                self.play_next_song()
            else:
                playback = self.spotify_client.current_playback()
                if not playback.get('is_playing', False):
                    print("Playback stopped unexpectedly; starting next song...")
                    self.play_next_song()
                else:
                    # When the song is nearly finished (0.5 seconds remaining), move on.
                    if self.time_to_wait <= 0.5:
                        time.sleep(self.time_to_wait)
                        self.time_to_wait = 0
                        print("Song nearly finished. Transitioning to next song...")
                        self.play_next_song()
                    else:
                        time.sleep(0.5)
                        self.time_to_wait -= 0.5


if __name__ == "__main__":
    spotify = Spotify_Manager()
    url = input("Enter Spotify playlist or album URL: ")
    # Start playback for the first time; the active source here is the URL entered.
    # (You may later add or remove sources via add_item_to_active_sources/remove_item_from_active_sources.)
    spotify.play_playlist_or_album(url, source=url)
    # Optionally, run the music handler loop.
    # For example, this could be run in a separate thread if sources are managed concurrently.
    spotify.handle_music()
