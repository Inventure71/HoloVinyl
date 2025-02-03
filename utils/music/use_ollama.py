import re

import ollama
import cv2
import time

def get_song_from_image(frame=None, max_retries=2):
    try:
        path = "custom_models/photo_of_object.png"
        retry_count = 0
        previous_error = ""
        image_description = "No image provided, invent something as if you took drugs"

        while retry_count <= max_retries:
            print(f"Attempt {retry_count + 1}/{max_retries + 1}")

            # Image capture and description remains the same
            if retry_count == 0:
                if frame is not None:
                    cv2.imwrite(path, frame)
                    time.sleep(0.1)

                messages = [
                    {
                        'role': 'user',
                        'content': 'Describe the main object in this picture in detail, the image contains at least an object on white background, ignore any hands.',
                        'images': [path]
                    }
                ]

                ollama.pull('moondream:1.8b')
                response = ollama.chat(model='moondream:1.8b', messages=messages)
                image_description = response.message.content
                print("Initial image description:", image_description)

            # Build system prompt with previous error if available
            system_prompt = """
**Objective:**  
You are given a description of an image (as a string). From that description, identify one primary object and produce a single line with five specific components separated by double pipes (`||`) in this order:  

1. **Musical Genre** (2 music terms): Suggest two music genres or styles that align with the object's mood and usage (e.g., "Jazz blues," "Synthwave electro," "Orchestral cinematic").  
2. **Sensation** (2 words): Describe the physical or emotional sensation the object might evoke in a person (e.g., "Warm comfort," "Sharp focus," "Playful joy").  
3. **Object** (2 words max): Identify the primary object in the image. Focus on tangible, inanimate objects (e.g., "Coffee mug," "Guitar," "Dragon statue"). Avoid body parts, vague terms, or abstract concepts.  
4. **Usage of the object** (3 words max): Describe how the object is most commonly used (e.g., "Morning caffeine," "Music performance," "Fantasy decoration").  
5. **Mood** (2 words): Convey the emotional tone or atmosphere associated with the object (e.g., "Relaxing vibe," "Energetic excitement," "Mysterious aura").  

**Final Output Format (all on one line, no extra text):**  
`musical genre||sensation||object||usage||mood`  

**Strict Rules:**  
1. **Formatting Violations Will Be Rejected:**  
   - Output must contain exactly 4 double-pipe separators (`||`).  
   - Output must follow the exact order: `musical genre||sensation||object||usage||mood`.  
   - No additional text, explanations, or sentences are allowed.  
   - Only ASCII characters are permitted.  

2. **Object Rules:**  
   - The object must be a tangible, inanimate item.  
   - Body parts, abstract concepts, or vague terms are not allowed.  
   - The object must be described in exactly 2 words or fewer.  

3. **Usage Rules:**  
   - Usage must be described in 3 words or fewer.  
   - Usage must directly relate to the object.  

4. **Mood Rules:**  
   - Mood must be described in exactly 2 words.  
   - Mood must align with the object's emotional tone.  

5. **Musical Genre Rules:**  
   - Exactly 2 music terms must be provided.  
   - Genres must align with the object's mood and usage.  

6. **Sensation Rules:**  
   - Sensation must be described in exactly 2 words.  
   - Sensation must describe a physical or emotional feeling.  

**Examples of Good Outputs:**  
- `Jazz blues||Warm comfort||Coffee mug||Morning caffeine||Relaxing vibe`  
- `Rock metal||Playful joy||Guitar||Music performance||Energetic excitement`  
- `Orchestral cinematic||Epic awe||Dragon statue||Fantasy decoration||Mysterious aura`  
- `Pop rock||Cool fizz||Soda can||Summer refreshment||Energetic mood`  
- `Ambient classical||Soft comfort||Book||Quiet reading||Calm focus`  

**Bad Examples (Rejected):**  
- `Various music||Enjoyment||Hand holding||Card game` (body parts are not objects)  
- `Fun||Dragon card||Social activity` (too vague, lacks sensation and musical genre)  
- `A pen might be used for studying or working, it can transmit a sensation of focus` (sentences are not allowed)  

**Additional Examples:**  
- `Ambient chill||Soft glow||Candle||Evening relaxation||Calm serenity`  
- `Indie pop||Wind rush||Bicycle||Urban commuting||Energetic freedom`  
- `Electronic downtempo||Focused clarity||Camera||Photo capturing||Creative inspiration`  
- `Smooth jazz||Elegant warmth||Wine glass||Evening toast||Romantic ambiance`  
- `Punk rock||Sharp thrill||Skateboard||Street tricks||Adrenaline rush`  

**Step-by-Step Evaluation Criteria:**  
1. **Musical Genre:** Are the two music terms appropriate for the object's mood and usage?  
2. **Sensation:** Does it describe a physical or emotional sensation in 2 words?  
3. **Object:** Is it a tangible, inanimate object? Is it described in 2 words or fewer?  
4. **Usage:** Is it concise (3 words max) and relevant to the object?  
5. **Mood:** Does it convey a clear emotional tone in 2 words?  
6. **Formatting:** Does the output strictly follow the `musical genre||sensation||object||usage||mood` format with exactly 4 double-pipe separators?  

**Common Pitfalls to Avoid:**  
- Too many/few words: Each component must adhere to the specified word count.  
- Extra commentary: The output line should contain only the 5 components separated by double pipes, with no additional explanation.  
- Incomplete or overly generic: Avoid vague terms like "Social activity" or using just a single word like "Happy" for mood.  
- Mentioning body parts or background: Focus on the object itself, not who is holding it or its environment.  
- Miscounting the double pipes: There must be exactly 4 instances of "||" to create 5 sections.  

**Summary:**  
Generate exactly one output line in the format:  
`musical genre||sensation||object||usage||mood`  
Note that the names are just placeholders and should be substituted with the corresponding values, nothing else should be included in the output.  """
            if previous_error:
                system_prompt += f"\n**CORRECT THIS FORMATTING ERROR FROM PREVIOUS ATTEMPT:**\n{previous_error}\n"

            messages = [
                {
                    'role': 'system',
                    'content': system_prompt + """
**Examples of Invalid/Incorrect Outputs:**  
- musical genre||sensation||object||usage||mood (missing components)                   
                    
**Examples of Valid Outputs:**  
- `Jazz blues||Warm comfort||Coffee mug||Morning caffeine||Relaxing vibe`  
- `Rock metal||Playful joy||Guitar||Music performance||Energetic excitement`  
- `Orchestral cinematic||Epic awe||Dragon statue||Fantasy decoration||Mysterious aura`  

**YOU MUST RESPOND WITH ONLY THE 5 COMPONENTS IN THE SPECIFIED FORMAT ON A SINGLE LINE. ANY OTHER TEXT WILL CAUSE ERRORS.**"""
                },
                {
                    'role': 'user',
                    'content': f"IMAGE DESCRIPTION: {image_description}" +
                               ("\n\nRE-EVALUATE WITH STRICT FORMATTING:" if retry_count > 0 else "")
                }
            ]

            # Get model response
            ollama.pull('deepseek-r1:1.5b')
            response = ollama.chat(
                model='deepseek-r1:1.5b',
                messages=messages,
                options={'temperature': 0.3}
            )

            try:
                print(f"\nResponse (Attempt {retry_count + 1}):", response.message.content)
                cleaned_text = re.sub(r"<think>.*?</think>\n*", "", response.message.content, flags=re.DOTALL)
                result = find_answer_from_response(cleaned_text)

                # Validate the result beyond just formatting
                if is_answer_valid(result):
                    return result.replace("||", " ").strip()
                else:
                    print("Invalid content despite correct formatting")
                    previous_error = f"LAST INVALID RESPONSE WAS:\n{response.message.content}\n\nREASON: Valid format but invalid content components. Make sure to follow the guidelines."

            except Exception as e:
                print("Parsing error:", e)
                previous_error = f"LAST INVALID RESPONSE WAS:\n{response.message.content}\n\nREASON: Failed format validation, make sure to follow the guidelines."

            retry_count += 1

        # Final fallback after all retries
        return cleaned_text

    except Exception as e:
        print("Critical error in get_song_from_image:", e)
        return None


def is_answer_valid(answer):
    components = answer.split('||')
    if len(components) != 5:
        return False

    # Check word counts
    word_counts = [
        (components[0], 5, "Musical Genre"),  # 5 music terms
        (components[1], 5, "Sensation"),
        (components[2], 5, "Object"),
        (components[3], 5, "Usage"),
        (components[4], 5, "Mood")
    ]

    for value, max_words, name in word_counts:
        words = value.strip().split()
        if len(words) > max_words:
            print(f"Invalid {name}: {value} (max {max_words} words)")
            return False

    # Check for placeholder terms
    forbidden_terms = ["musical genre", "sensation", "object", "usage", "mood"]
    for term in forbidden_terms:
        if term in answer.lower():
            print(f"Invalid placeholder term found: {term}")
            return False

    return True


# Modified find_answer_from_response
def find_answer_from_response(cleaned_text):
    forbidden_components = ["musical genre", "sensation", "object", "usage", "mood"]

    # First pass: Strict validation
    for line in reversed(cleaned_text.split('\n')):
        line = line.strip()
        if line.count('||') == 4:
            parts = line.split('||')
            if len(parts) != 5:
                continue

            # Check for header line
            if all(part.strip().lower() in forbidden_components for part in parts):
                continue

            # Check for partial headers
            forbidden_count = sum(
                1 for component in forbidden_components if any(component in part.lower() for part in parts))
            if forbidden_count > 1:
                continue

            return line

    # Second pass: Lenient validation
    for line in reversed(cleaned_text.split('\n')):
        line = line.strip()
        if line.count('||') == 4:
            return line

    # Final fallback: Return first line with pipes
    for line in cleaned_text.split('\n'):
        if '||' in line:
            return line

    return cleaned_text