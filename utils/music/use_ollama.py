import time

import cv2
import ollama
import re


def get_song_from_image(frame=None):
    try:
        path = "custom_models/photo_of_object.png"

        print("Getting song from image")
        if frame is not None:
            cv2.imwrite(path, frame)
            time.sleep(0.1)

        print("Image saved")
        messages = [
            {
                'role': 'user',
                'content': 'Describe the main object in this picture in detail, the image contains at least an object on white background.',
                'images': [path]
            }
        ]

        ollama.pull('moondream:1.8b')
        response = ollama.chat(
            model='moondream:1.8b',
            messages=messages,
        )
        print("Image description", response.message.content)

        #messages.append({
        #    'role': 'assistant',
        #    'content': response.message.content
        #})

        messages = [
            {
                'role': 'user',
                'content': "Analyze the described object to identify: \n"
                           "1. Primary activity associated with the object\n"
                           "2. Core theme it represents\n"
                           "3. Matching music genre/situation\n\n"
                           "Consider cultural associations and usage context. Examples:\n"
                           "- Pen → Studying, Focus, Instrumental/Classical\n"
                           "- Pizza → Cooking, Celebration, Upbeat Italian\n"
                           "- Running shoes → Exercise, Energy, Pump-up Playlist\n\n"
                           "Respond ONLY with 3 comma-separated values in this format:"
                           "Activity, Theme, Genre. No explanations.\n"
                           f"{response.message.content}",
            }
        ]

        ollama.pull('deepseek-r1:1.5b')  # deepseek-r1:1.5b or qwen2.5:0.5b
        response = ollama.chat(
            model='deepseek-r1:1.5b',  # also have deepseeker small
            messages=messages,
        )

        #print(response.message.content)

        cleaned_text = re.sub(r"<think>.*?</think>\n*", "", response.message.content, flags=re.DOTALL)
        # Remove brackets, parentheses, and curly braces
        cleaned_text = re.sub(r"[\[\]{}()]", "", cleaned_text)
        # Remove all newlines and extra spaces, then strip leading/trailing spaces
        cleaned_text = " ".join(cleaned_text.split()).strip()

        print(cleaned_text)
        return cleaned_text
    except Exception as e:
        print("Error in get_song_from_image")
        print(f"Error: {e}")
        return None


