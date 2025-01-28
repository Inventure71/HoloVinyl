import json
import os

from utils.pygame_utils.Button import Button
from utils.pygame_utils.TextField import TextField

# File to save mappings
MAPPING_FILE = "class_mappings.json"

def load_mappings():
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "r") as f:
            return json.load(f)
    return {}

def save_mappings(mappings):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mappings, f, indent=4)

class Submenu:
    def __init__(self, screen, font, yolo_handler):
        self.screen = screen
        self.font = font
        self.yolo_handler = yolo_handler
        self.mappings = load_mappings()
        self.classes = self.yolo_handler.get_classes()
        self.text_fields = []
        self.buttons = []
        self.create_ui()
        self.active = False

    def create_ui(self):
        self.text_fields = []
        self.buttons = []

        y_offset = 100
        for class_name in self.classes:
            text_field = TextField(
                200, y_offset, 200, 40, self.font, text_color=(0, 0, 0), bg_color=(255, 255, 255), border_color=(0, 0, 0),
            )
            text_field.text = self.mappings.get(class_name, "")
            self.text_fields.append((class_name, text_field))
            y_offset += 60

        save_button = Button(
            x=400, y=y_offset, width=150, height=50, text="Save", font=self.font, text_color=(255, 255, 255),
            button_color=(0, 128, 255), hover_color=(0, 102, 204), callback=self.save_mappings
        )
        self.buttons.append(save_button)

    def save_mappings(self):
        for class_name, text_field in self.text_fields:
            self.mappings[class_name] = text_field.text
        save_mappings(self.mappings)
        print("Mappings saved successfully!")

    def draw(self):
        self.screen.fill((50, 50, 50))
        title = self.font.render("Class Mappings", True, (255, 255, 255))
        self.screen.blit(title, (300, 50))

        for class_name, text_field in self.text_fields:
            label = self.font.render(class_name, True, (255, 255, 255))
            self.screen.blit(label, (50, text_field.rect.y + 10))
            text_field.draw(self.screen)

            text_field.update()

        for button in self.buttons:
            button.draw(self.screen)

    def handle_event(self, event):
        for _, text_field in self.text_fields:
            text_field.handle_event(event)

        for button in self.buttons:
            button.handle_event(event)