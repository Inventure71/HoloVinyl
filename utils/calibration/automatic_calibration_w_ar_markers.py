import os
import random
import cv2
import numpy as np


class ArMarkerHandler:
    def __init__(self):
        # Load the ArUco dictionary (DICT_4X4_50 is a good small marker set)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def create_board(self, output_path="custom_models/board.png", margin=50):
        """Creates an A4-sized board with four ArUco markers forming a perfect square,
        with the left-side markers fixed near the A4 corners and the right-side markers forming a square.
        """

        # Define A4 size in pixels at 300 DPI (Landscape mode)
        a4_width, a4_height = 3508, 2480
        marker_size = 150  # Marker size in pixels

        # Create a white canvas
        board = np.ones((a4_height, a4_width), dtype=np.uint8) * 255

        # **Fixed left-side markers near the A4 corners with margin**
        top_left_x, top_left_y = margin, margin  # Top-left corner with margin
        bottom_left_x, bottom_left_y = margin, a4_height - marker_size - margin  # Bottom-left with margin

        # **Calculate the square size**
        square_size = bottom_left_y - top_left_y  # The vertical distance between top-left and bottom-left

        # **Ensure square doesn't go out of bounds**
        square_size = min(square_size, a4_width - 2 * margin)

        # **Compute right-side marker positions to ensure a perfect square**
        top_right_x = top_left_x + square_size
        top_right_y = top_left_y

        bottom_right_x = bottom_left_x + square_size
        bottom_right_y = bottom_left_y

        # **Define marker positions in a perfect square**
        corners = [
            (top_left_x, top_left_y),  # Top-left (fixed)
            (top_right_x, top_right_y),  # Top-right (adjusted for square)
            (bottom_left_x, bottom_left_y),  # Bottom-left (fixed)
            (bottom_right_x, bottom_right_y)  # Bottom-right (adjusted for square)
        ]

        # **Generate and place markers**
        for i, (x, y) in enumerate(corners):
            marker_id = i  # Assign marker IDs 0 to 3
            marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size, marker_img, 1)

            # Place the marker in the board
            board[y:y + marker_size, x:x + marker_size] = marker_img

        # **Save the final board image**
        cv2.imwrite(output_path, board)
        print(f"Board saved at {output_path}")

    def detect_corners(self, input_image, use_webcam=False):
        """
        Detects the four ArUco markers in the input image and returns their pixel coordinates.
        """
        if use_webcam:
            image = input_image
        else:
            image = cv2.imread(input_image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None or len(ids) < 4:
            print("Not all four markers were detected!")
            return None

        # Map detected IDs to their corners
        marker_dict = {int(ids[i]): corners[i][0] for i in range(len(ids))}

        # Ensure we have all 4 markers
        if set(marker_dict.keys()) != {0, 1, 2, 3}:
            print(f"Markers detected: {list(marker_dict.keys())}, expected [0,1,2,3]")
            return None

        # Sort markers based on expected positions (TL, TR, BL, BR)
        sorted_corners = np.array([
            marker_dict[0][0],  # Top-left
            marker_dict[1][1],  # Top-right
            marker_dict[2][3],  # Bottom-left
            marker_dict[3][2],  # Bottom-right
        ], dtype=np.float32)

        print("Corrected Corners Detected:", sorted_corners)
        return sorted_corners

    def warp_and_crop_board(self, image, corners=None, is_for_frame=False):
        if corners is None:
            print("No corners provided! Run detect_corners() first.")
            return None

        # Ensure corners is a NumPy float32 array
        corners = np.array(corners, dtype=np.float32)

        # Get frame size
        height, width = image.shape[:2]

        # Automatically adjust size based on corners
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)
        width_out = int(x_max - x_min)
        height_out = int(y_max - y_min)

        # Define destination points for warping
        dst_pts = np.array([
            [0, 0],  # Top-left
            [width_out, 0],  # Top-right
            [0, height_out],  # Bottom-left
            [width_out, height_out]  # Bottom-right
        ], dtype=np.float32)

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width_out, height_out))

        warped = cv2.resize(warped, (640, 640))
        cv2.imwrite("../../custom_models/markers/board_warped.png", warped)

        return warped

    # REDUNDANT FUNCTION
    # TODO: Remove this function and implement in one above
    def warp_and_adjust(self, image, corners=None):
        if corners is None:
            print("No corners provided! Run detect_corners() first.")
            return None

        # Ensure corners is a NumPy float32 array
        corners = np.array(corners, dtype=np.float32)

        # Get frame size
        height, width = image.shape[:2]

        # Get bounding box of the detected corners
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)

        # Calculate the largest possible rectangle without going out of bounds
        width_out = min(int(x_max - x_min), width)
        height_out = min(int(y_max - y_min), height)

        # Define destination points for warping
        dst_pts = np.array([
            [x_min, y_min],  # Top-left
            [x_min + width_out, y_min],  # Top-right
            [x_min, y_min + height_out],  # Bottom-left
            [x_min + width_out, y_min + height_out]  # Bottom-right
        ], dtype=np.float32)

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (width, height))

        # Resize for uniformity (optional)
        warped = cv2.resize(warped, (640, 640))

        # Save the adjusted warped image
        cv2.imwrite("../../custom_models/markers/board_warped_adjusted.png", warped)

        return warped

"""
# Example Usage
marker_handler = ArMarkerHandler()
marker_handler.create_board()

# Detect corners first
detected_corners = marker_handler.detect_corners()

# Warp and crop using the detected corners
if detected_corners is not None:
    marker_handler.warp_and_crop_board(corners=detected_corners)"""