import os
import time

import cv2
import pygame

from UI import UI
from utils.database_handler_V3 import create_or_update_yolo_dataset
from utils.string_utils import unsanitize_string, sanitize_string

#TODO: added sanitization of strings, removing spaces ecc. Check if it works

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
    new_class = sanitize_string(ui.adding_class)
    os.makedirs(f"custom_models/raw_images/{new_class}", exist_ok=True)

    if ui.remaining_photo_count > 0:
        ui.remaining_photo_count -= 1
        cv2.imwrite(f"custom_models/raw_images/{new_class}/img_{len(os.listdir(f'custom_models/raw_images/{new_class}'))}.png", frame)

    if ui.remaining_photo_count <= 0:
        ui.adding_class = ""
        print("Finished adding class!")

def button_clicked_train_model():
    print("Button Train!")

    start_time = time.time()

    directories = os.listdir("custom_models/raw_images")
    already_existing_classes = ui.yolo_handler.get_classes()
    new_classes_dirs = {}

    for directory in directories:
        if unsanitize_string(directory) not in already_existing_classes:
            print(f"Adding class: {directory}")
            new_classes_dirs[directory] = f"custom_models/raw_images/{directory}"

    print(f"Missing classes: {new_classes_dirs}")

        #new_class_dirs = {
    #    "happy face": "raw_images/happy_face",
    #    "plane": "raw_images/plane"
    #}

    create_or_update_yolo_dataset(
        class_directories=new_classes_dirs,
        output_directory="custom_models/yolo_dataset",
        target_samples_per_class=50,
        debug_boundaries=False,
        #existing_dataset="custom_models/yolo_dataset"
    )

    print(f"Dataset creation completed in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()

    # Train on a custom dataset
    ui.yolo_handler.train_model(
        data_path="custom_models/yolo_dataset/dataset.yaml",
        model_type="yolo11n.pt",  # Small model
        epochs=50,
        batch_size=16,
        img_size=600,
        save_dir="custom_models/runs/train"
    )
    print(f"Model training completed in {time.time() - start_time:.2f} seconds.")

    # TEST: manually move runs folder under custom_models
    print("Manually moving runs under custom_models")
    os.system("mv runs custom_models")
    print("Moved folder and completed training")

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
    points = [(0, 0), (600, 0), (600, 600), (0, 600)]
    enable_spotify = False

    ui = UI(points, enable_spotify, button_clicked_start_prediction, button_clicked_add_class, button_clicked_train_model, button_clicked_open_submenu, button_clicked_quit, button_clicked_take_photo)
    pygame.scrap.init()
    ui.run()