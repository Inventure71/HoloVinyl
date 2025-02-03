import playsound


def play_sound(sound_path="utils/music/sounds/SuccessfullyClicked.mp3"):
    playsound.playsound(sound_path)
    print("Sound played")
