import os
import time

import cv2
import pygame

from UI import UI
from utils.database_handler_V3 import create_or_update_yolo_dataset

"""BUTTONS START"""
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
"""BUTTONS END"""


if __name__ == "__main__":
    pygame.init()
    #calibration = ManualBoardCalibration()
    #points = calibration.run()
    points = [(0, 0), (640, 0), (640, 480), (0, 480)]

    ui = UI(points, button_clicked_start_prediction, button_clicked_add_class, button_clicked_take_photo, button_clicked_train_model, button_clicked_quit, button_clicked_open_submenu)
    pygame.scrap.init()
    ui.run()