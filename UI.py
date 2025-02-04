import json
import os
import threading

import cv2
import numpy as np
import pygame

from HandTracking import HandTrackingManager
from utils.calibration.automatic_calibration_w_ar_markers import ArMarkerHandler
from utils.generic import load_mappings
from utils.music.sounds_utils import play_sound
from utils.pygame_utils.Button import Button
from utils.pygame_utils.TextField import TextField
from submenu_UI import Submenu
from utils.music.spotify_manager import Spotify_Manager
from utils.yolo_handler import YOLOHandler

from addButtonUI import DigitalButtonEditor



class UI:
    def __init__(self, camera, points, enable_spotify, button_clicked_start_prediction, button_clicked_add_class, button_clicked_train_model, button_clicked_open_submenu, button_clicked_quit, button_clicked_take_photo):
        self.screen = pygame.display.set_mode((1024, 640))
        pygame.display.set_caption("UI TEST")
        self.clock = pygame.time.Clock()
        self.running = True

        self.webcam = cv2.VideoCapture(camera)
        _, self.frame = self.webcam.read()

        self.calibration_active = True if points is None else False
        self.marker_handler = ArMarkerHandler()

        if self.calibration_active:
            self.calibration_points = None
            self.calibrate_board(self.frame)
        else:
            self.calibration_points = points

        self.predicting = False
        self.adding_class = ""
        self.remaining_photo_count = 0

        # init font
        self.font = pygame.font.Font(None, 32)

        # list of 4 buttons
        self.buttons = [
            Button(
                x=600-50, #
                y=0,
                width=50,
                height=50,
                text="TP",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_start_prediction,
            ),
            Button(
                x=1024-200,
                y=0,
                width=200,
                height=50,
                text="Add Class",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_add_class,
            ),
            Button(
                x=1024-200,
                y=80,
                width=200,
                height=50,
                text="Train Model",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_train_model,
            ),
            Button(
                x=1024 - 200, y=160, width=200, height=50, text="Class Mappings", font=self.font,
                text_color=(255, 255, 255), button_color=(0, 128, 255), hover_color=(0, 102, 204),
                callback=button_clicked_open_submenu),
            Button(
                x=1024-200,
                y=550,
                width=200,
                height=50,
                text="Quit",
                font=self.font,
                text_color=(255, 255, 255),
                button_color=(0, 128, 255),
                hover_color=(0, 102, 204),
                callback=button_clicked_quit,
            ),
        ]

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

        self.text_field = TextField(1024-200-200, 0, 200, 50, self.font, text_color=(0, 0, 0), bg_color=(255, 255, 255), border_color=(0, 0, 0))


        self.yolo_handler = YOLOHandler(model_path="yolo11n.pt")
        self.reload_YOLO_model()

        self.submenu = Submenu(self.screen, self.font, self.yolo_handler)

        self.training_in_progress = False

        self.hand_tracking_manager = HandTrackingManager(callback_function=self.user_pinched)
        self.radius_of_click = 50

        """MUSIC RELATED"""
        self.paused = False
        self.mappings = load_mappings()  # TODO update mappings once submenu is closed
        self.active_sources = []
        self.class_frame_count = {}  # Tracks consecutive frames for each class
        self.threshold_frames = 5  # Number of consecutive frames needed to add to the queue
        self.count_in_a_row = {}
        self.enable_spotify = enable_spotify # variable to check if i should activate it
        if self.enable_spotify:
            self.spotify_manager = Spotify_Manager()
            # Run handle_music function in a separate thread
            self.music_thread = threading.Thread(target=self.spotify_manager.handle_music,
                                                 daemon=True)  # `daemon=True` allows the thread to exit when the main program exits
            self.music_thread.start()


        """HAND TRACKING"""
        self.digital_button_ui = None
        self.draw_buttons = []
        self.selecting_buttons_UI_active = True

    def button_clicked(self, n):
        print(f"Button {n} clicked - INSIDE CLASS")
        if n == 0 and self.enable_spotify:
            play_sound()
            self.spotify_manager.play_pause()
        if n == 1 and self.enable_spotify:
            play_sound()
            self.spotify_manager.find_song_based_on_image(self.frame)


    def user_pinched(self, mouse_position):
        print("mouse clicked", mouse_position)
        for button in self.draw_buttons:
            # Only call this button's callback if the pinch is within its rectangle.
            if button.rect.collidepoint(mouse_position):
                print(f"Called button {button.text}")
                button.callback()
                return True

        """
                # When processing clicks:
                #cropped_point =  self.marker_handler.map_click_to_cropped_space(
                #    click_point=mouse_position
                #)

                if cropped_point is not None:
                    print(f"Clicked on cropped space: {cropped_point}")
                else:
                    print("Clicked outside of cropped space")"""

        #self.hand_tracking_manager.is_pinching

    def calibrate_board(self, frame):
        if self.calibration_active:
            self.calibration_points = self.marker_handler.detect_corners(frame, use_webcam=True)
            if self.calibration_points is not None:
                print("Calibration successful!")
                return
            print("Calibration failed!")

        print("Calibration not active!")
        # Convert corners list to a NumPy array
        self.calibration_points = np.array([(0, 0), (640, 0), (640, 640), (0, 640)], dtype=np.float32)
        return

    def reload_YOLO_model(self, custom = True):
        if custom:
            self.yolo_handler.load_model(model_path="custom_models/runs/detect/train/weights/best.pt")

        else:
            self.yolo_handler.load_model(model_path="yolo11n.pt")

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))

    def detect_active_sources(self, detected_classes):
        for cls in detected_classes:
            self.count_in_a_row[cls] = self.count_in_a_row.get(cls, 0) + 1

            url = self.mappings.get(cls, '')
            if url == '':
                print(f"Class {cls} not found OR empty in mappings, skipping...")
                continue

            if self.count_in_a_row[cls] >= self.threshold_frames and url not in self.active_sources:
                self.count_in_a_row[cls] = self.threshold_frames - 1 # To limit the number of frames in a row
                self.active_sources.append(url)
                self.spotify_manager.add_item_to_active_sources(url)
                print(f"Added to active sources: {url}", "Active sources:", self.active_sources)
            elif self.count_in_a_row[cls] >= self.threshold_frames:
                self.count_in_a_row[cls] = self.threshold_frames - 1 # To limit the number of frames in a row


        for cls in list(self.count_in_a_row.keys()):
            if cls not in detected_classes:
                self.count_in_a_row[cls] -= 1
                print(f"Class {cls} not detected, count decreased to {self.count_in_a_row[cls]}")
                if self.count_in_a_row[cls] <= 0:
                    print("Item not seen for a while")
                    del self.count_in_a_row[cls]

                    url = self.mappings.get(cls, '')
                    if url == '':
                        print(f"Class {cls} not found OR empty in mappings, skipping...")
                        continue

                    if url in self.active_sources:
                        self.spotify_manager.remove_item_from_active_sources(url)
                        self.active_sources.remove(url)
                        print(f"Removed from active sources: {url}", "Active sources:", self.active_sources)
                    else:
                        print(f"Class {cls} not in active sources, skipping...")

    def process_frame(self, frame, frame_timestamp_ms):
        """
        Process a single video frame and display predictions.
        :param frame: The current video frame.
        """
        # Get predictions for the frame
        predictions = self.yolo_handler.predict(frame, conf_threshold=0.5, save=False, save_dir="custom_models/runs/predict")
        detected_classes = set()

        try:
            self.display_frame(frame)
            for pred in predictions[0]:  # Assuming predictions for one frame
                x1, y1, x2, y2 = [int(coord) for coord in pred["box"]]
                label = pred["label"]
                confidence = pred["confidence"]
                print("Prediction:", label, confidence)

                detected_classes.add(label)

                # Draw bounding box
                pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

                # Draw label and confidence
                font = pygame.font.Font(None, 24)
                text = font.render(f"{label} ({confidence:.2f})", True, (144, 238, 144))
                self.screen.blit(text, (x1, y1 - 20))  # Above the box

            if self.enable_spotify and (frame_timestamp_ms % 1 == 0):
                self.detect_active_sources(detected_classes)
        except Exception as e:
            print("Error while processing predictions:", e)

    def run(self):
        frame_timestamp_ms = 0
        ret, frame = self.webcam.read()

        # Load the digital button UI
        if self.digital_button_ui is None:
            self.digital_button_ui = DigitalButtonEditor(background_frame=frame, callback=self.button_clicked)
            self.draw_buttons = self.digital_button_ui.buttons

        while self.running:
            if self.selecting_buttons_UI_active:
                if self.digital_button_ui is None:
                    self.digital_button_ui = DigitalButtonEditor(background_frame=frame, callback=self.button_clicked)

                self.selecting_buttons_UI_active, self.draw_buttons = self.digital_button_ui.run(self.screen, self.clock)

            else:
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

                # TODO: check if i can just crop frame_temp to 600*600 instead of running warp and crop

                # new version
                frame_temp = self.marker_handler.warp_and_adjust(frame, corners=self.calibration_points)
                self.hand_tracking_manager.analyze_frame(frame_temp, frame_timestamp_ms)
                frame_timestamp_ms += 1

                frame = self.marker_handler.warp_and_crop_board(frame, corners=self.calibration_points, is_for_frame=True)
                #frame = transform_to_square(frame, self.calibration_points)


                self.frame = frame
                #frame = cv2.resize(frame, (600, frame.shape[0] * 600 // frame.shape[1]))

                if self.adding_class != "":
                    class_name = self.text_field.text
                    print(f"Adding class: {class_name}")
                    directory = f"custom_models/raw_images/{class_name}"
                    os.makedirs(directory, exist_ok=True)

                    self.display_frame(frame)

                    self.button_to_take_picture.draw(self.screen)
                    self.screen.blit(self.font.render(f"Remaining photos: {self.remaining_photo_count}", True, (255, 255, 255)), (200, 400))

                elif self.predicting:
                    self.screen.blit(self.font.render("Predicting...", True, (255, 255, 255)), (200, 400))

                    self.process_frame(frame, frame_timestamp_ms)

                else:
                    self.display_frame(frame)

                if self.submenu.active:
                    self.submenu.draw()

                else:
                    for button in self.buttons:
                        button.draw(self.screen)

                    #for button in self.draw_buttons:
                    #    button.draw(self.screen)

                    # Update the text field
                    self.text_field.update()

                    # Draw the text field
                    self.text_field.draw(self.screen)

            self.clock.tick(30)
            pygame.display.flip()

        self.webcam.release()
        pygame.quit()


"""
if __name__ == "__main__":
    pygame.init()
    calibration = ManualBoardCalibration()
    points = calibration.run()
    points = [(0, 0), (640, 0), (640, 480), (0, 480)]

    ui = UI(points)
    pygame.scrap.init()
    ui.run()"""