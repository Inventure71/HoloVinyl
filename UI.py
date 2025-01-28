import os
import time

import cv2
import numpy as np
import pygame

from pygame_utils.Button import Button
from pygame_utils.TextField import TextField
from submenu_UI import Submenu
from utils.board_calibration import ManualBoardCalibration

from utils.database_handler_V3 import create_or_update_yolo_dataset
from utils.image_utils import transform_to_square
from utils.yolo_handler import YOLOHandler


def button_clicked_start_prediction():
    print("Button Start Prediction!")
    ui.predicting = not ui.predicting

def button_clicked_add_class():
    # just add to dataset
    print("Button Add Class!")
    ui.remaining_photo_count = 5
    ui.adding_class = ui.text_field.text

def button_clicked_take_photo(frame):
    print("Button Took Picture!")
    os.makedirs(f"raw_images/{ui.adding_class}", exist_ok=True)

    if ui.remaining_photo_count > 0:
        ui.remaining_photo_count -= 1
        cv2.imwrite(f"raw_images/{ui.adding_class}/img_{len(os.listdir(f'raw_images/{ui.adding_class}'))}.png", frame)

    if ui.remaining_photo_count <= 0:
        ui.adding_class = ""
        print("Finished adding class!")

def button_clicked_train_model():
    print("Button Train!")

    start_time = time.time()

    # TODO: make this automatic
    new_class_dirs = {
        "happy face": "raw_images/happy_face",
        "plane": "raw_images/plane"
    }
    new_class_dirs = {
        "amongus": "raw_images/amongus",
    }

    create_or_update_yolo_dataset(
        class_directories=new_class_dirs,
        output_directory="yolo_dataset",
        target_samples_per_class=70,
        existing_dataset="yolo_dataset"
    )

    print(f"Dataset creation completed in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()

    # Train on a custom dataset
    ui.yolo_handler.train_model(
        data_path="yolo_dataset/dataset.yaml",
        model_type="yolo11n.pt",  # Small model
        epochs=50,
        batch_size=16,
        img_size=640
    )
    print(f"Model training completed in {time.time() - start_time:.2f} seconds.")

def button_clicked_quit():
    print("Button Quit!")
    ui.running = False

def button_clicked_open_submenu():
    ui.submenu.active = True

class UI:
    def __init__(self, points):
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("UI TEST")
        self.clock = pygame.time.Clock()
        self.running = True

        self.calibration_points = points

        self.webcam = cv2.VideoCapture(4)

        self.predicting = False
        self.adding_class = ""
        self.remaining_photo_count = 0

        # init font
        self.font = pygame.font.Font(None, 32)

        # list of 4 buttons
        self.buttons = [
            Button(
                x=1024-150,
                y=120,
                width=150,
                height=50,
                text="Toggle Prediction",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_start_prediction,
            ),
            Button(
                x=1024-150,
                y=200,
                width=150,
                height=50,
                text="Add Class",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_add_class,
            ),
            Button(
                x=1024-150,
                y=280,
                width=150,
                height=50,
                text="Train Model",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_train_model,
            ),
            Button(
                x=1024 - 150, y=360, width=150, height=50, text="Class Mappings", font=self.font,
                text_color=(255, 255, 255), button_color=(0, 128, 255), hover_color=(0, 102, 204),
                callback=button_clicked_open_submenu),
            Button(
                x=1024-150,
                y=440,
                width=150,
                height=50,
                text="Quit",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_quit,
            ),
        ]

        self.frame = None
        self.button_to_take_picture = Button(
                    x=512,
                    y=440,
                    width=150,
                    height=50,
                    text="Take Picture",
                    font=self.font,
                    text_color=(255, 255, 255),
                    button_color=(0, 128, 255),
                    hover_color=(0, 102, 204),
                    callback= lambda: button_clicked_take_photo(self.frame),
                )

        self.text_field = TextField(0, 500, 400, 50, self.font, text_color=(0, 0, 0), bg_color=(255, 255, 255), border_color=(0, 0, 0))


        self.yolo_handler = YOLOHandler(model_path="yolo11n.pt")
        self.reload_YOLO_model()

        self.submenu = Submenu(self.screen, self.font, self.yolo_handler)



    def reload_YOLO_model(self, custom = True):
        if custom:
            self.yolo_handler.load_model(model_path="runs/detect/train3/weights/best.pt")

        else:
            self.yolo_handler.load_model(model_path="yolo11n.pt")

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))

    def process_frame(self, frame):
        """
        Process a single video frame and display predictions.
        :param frame: The current video frame.
        """
        # Get predictions for the frame
        predictions = self.yolo_handler.predict(frame, conf_threshold=0.5, save=False, save_dir="./runs/predict")

        try:
            self.display_frame(frame)
            for pred in predictions[0]:  # Assuming predictions for one frame
                print("Prediction:", pred)
                x1, y1, x2, y2 = [int(coord) for coord in pred["box"]]
                label = pred["label"]
                confidence = pred["confidence"]

                # Draw bounding box
                pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

                # Draw label and confidence
                font = pygame.font.Font(None, 24)
                text = font.render(f"{label} ({confidence:.2f})", True, (255, 255, 255))
                self.screen.blit(text, (x1, y1 - 20))  # Above the box
        except Exception as e:
            print("Error while processing predictions:", e)

    def run(self):
        while self.running:
            self.screen.fill((0,0,0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if self.submenu.active:
                    self.submenu.handle_event(event)

                else:
                    for button in self.buttons:
                        button.handle_event(event)

                    if self.adding_class != "":
                        self.button_to_take_picture.handle_event(event)

                    self.text_field.handle_event(event)

            ret, frame = self.webcam.read()
            frame = transform_to_square(frame, self.calibration_points)
            self.frame = frame
            frame = cv2.resize(frame, (600, frame.shape[0] * 600 // frame.shape[1]))

            if self.adding_class != "":
                class_name = self.text_field.text
                print(f"Adding class: {class_name}")
                directory = f"raw_images/{class_name}"
                os.makedirs(directory, exist_ok=True)

                self.display_frame(frame)

                self.button_to_take_picture.draw(self.screen)
                self.screen.blit(self.font.render(f"Remaining photos: {self.remaining_photo_count}", True, (255, 255, 255)), (200, 400))

            elif self.predicting:
                self.screen.blit(self.font.render("Predicting...", True, (255, 255, 255)), (200, 400))

                self.process_frame(frame)

            else:
                self.display_frame(frame)

            if self.submenu.active:
                self.submenu.draw()


            else:
                for button in self.buttons:
                    button.draw(self.screen)

                # Update the text field
                self.text_field.update()

                # Draw the text field
                self.text_field.draw(self.screen)

            self.clock.tick(60)
            pygame.display.flip()

        self.webcam.release()
        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    #calibration = ManualBoardCalibration()
    #points = calibration.run()
    points = [(0, 0), (640, 0), (640, 480), (0, 480)]

    ui = UI(points)
    pygame.scrap.init()
    ui.run()