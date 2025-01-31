import cv2
import ollama
import re


def get_song_from_image(frame):
    try:
        path = "image.pngcustom_models/photo_of_object.png"

        if frame:
            cv2.imwrite(path, frame)

        messages = [
            {
                'role': 'user',
                'content': 'Describe the main object in this picture in detail',
                'images': [path]
            }
        ]

        ollama.pull('moondream:1.8b')
        response = ollama.chat(
            model='moondream:1.8b',
            messages=messages,
        )
        #print(response.message.content)

        messages.append({
            'role': 'assistant',
            'content': response.message.content
        })

        messages.append({
            'role': 'user',
            'content': "Extract mood, theme, and keywords from the image description for finding a song.\n"
                       "The output should just be a list of Mood, themes and keywords with no explanation for any of them, all in a line, no indentation"
            # 'Describe it in 3 words',
        })

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
        print(f"Error: {e}")
        return None


