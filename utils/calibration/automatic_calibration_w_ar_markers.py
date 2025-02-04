import os
import random
import cv2
import numpy as np


class ArMarkerHandler:
    def __init__(self):
        # Load the ArUco dictionary (DICT_4X4_50 is a good small marker set)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.original_size_wc = None
        self.matrix_wa = None
        self.matrix_wc = None
        self.warped_corners = None

        # Cache for warp_and_crop_board
        self.last_corners_wc = None
        self.last_matrix_wc = None
        self.last_original_size_wc = None

        # Cache for warp_and_adjust
        self.last_corners_wa = None
        self.last_adjusted_matrix_wa = None
        self.last_output_width_wa = None
        self.last_output_height_wa = None
        self.last_warped_corners_wa = None


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

        corners = np.array(corners, dtype=np.float32)

        # Check if we can reuse cached values
        if (self.last_corners_wc is not None and
                np.allclose(corners, self.last_corners_wc, atol=1e-3)):
            matrix = self.last_matrix_wc
            width_out, height_out = self.last_original_size_wc
        else:
            # Calculate new transformation values
            x_min, y_min = np.min(corners, axis=0)
            x_max, y_max = np.max(corners, axis=0)
            width_out = int(x_max - x_min)
            height_out = int(y_max - y_min)

            dst_pts = np.array([
                [0, 0], [width_out, 0],
                [0, height_out], [width_out, height_out]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(corners, dst_pts)

            # Update cache
            self.last_corners_wc = corners.copy()
            self.last_matrix_wc = matrix.copy()
            self.last_original_size_wc = (width_out, height_out)
            self.matrix_wc = matrix
            self.original_size_wc = (width_out, height_out)

        # Perform warping with current image
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

        corners = np.array(corners, dtype=np.float32)

        # Check if we can reuse cached values
        if (self.last_corners_wa is not None and
            np.allclose(corners, self.last_corners_wa, atol=1e-3)):
            adjusted_matrix = self.last_adjusted_matrix_wa
            output_width = self.last_output_width_wa
            output_height = self.last_output_height_wa
            warped_corners = self.last_warped_corners_wa.copy()
        else:
            # Calculate new transformation values
            top_left, top_right, bottom_left, bottom_right = corners
            width_top = np.linalg.norm(top_right - top_left)
            width_bottom = np.linalg.norm(bottom_right - bottom_left)
            width = int((width_top + width_bottom) / 2)

            height_left = np.linalg.norm(bottom_left - top_left)
            height_right = np.linalg.norm(bottom_right - top_right)
            height = int((height_left + height_right) / 2)

            dst_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(corners, dst_pts)

            # Calculate output bounds
            h, w = image.shape[:2]
            orig_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
            transformed_corners = cv2.perspectiveTransform(orig_corners.reshape(1, -1, 2), matrix).reshape(-1, 2)

            min_x, min_y = np.min(transformed_corners, axis=0)
            max_x, max_y = np.max(transformed_corners, axis=0)
            output_width = int(np.ceil(max_x - min_x))
            output_height = int(np.ceil(max_y - min_y))

            # Adjust matrix
            translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
            adjusted_matrix = translation @ matrix

            # Calculate warped corners
            warped_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), adjusted_matrix).reshape(-1, 2)

            # Update cache
            self.last_corners_wa = corners.copy()
            self.last_adjusted_matrix_wa = adjusted_matrix.copy()
            self.last_output_width_wa = output_width
            self.last_output_height_wa = output_height
            self.last_warped_corners_wa = warped_corners.copy()
            self.matrix_wa = matrix
            self.warped_corners = warped_corners

        # Perform warping with current image
        warped = cv2.warpPerspective(image, adjusted_matrix, (output_width, output_height))
        cv2.imwrite("../../custom_models/markers/board_warped.png", warped)
        return warped

    def map_click_to_cropped_space(self, click_point):
        """
        Map a click point from the warp_and_adjust coordinate system to the 640x640
        coordinate system of the warp_and_crop_board output.

        This approach assumes that:
          - self.warped_corners is available (set in warp_and_adjust)
          - self.warped_corners contains the four board corners (in any order; we will
            compute the top-left and bottom-right)
          - warp_and_crop_board uses the original corners to define a board of size
            (width_out, height_out) which is then resized to 640x640. This width and height
            are stored in self.original_size_wc.

        Args:
            click_point: A tuple (x, y) representing the click in warp_and_adjust space.

        Returns:
            A tuple (mapped_x, mapped_y) in the 640x640 coordinate space, or None if the click
            falls outside the board.
        """
        if not hasattr(self, 'warped_corners'):
            print("Error: warped_corners not available. Run warp_and_adjust first.")
            return None
        if not hasattr(self, 'original_size_wc'):
            print("Error: original board size not available. Run warp_and_crop_board first.")
            return None

        # Compute the board's bounding box in warp_and_adjust space.
        warped_corners = self.warped_corners  # assumed shape (4, 2)
        board_top_left = np.min(warped_corners, axis=0)  # [x_min, y_min]
        board_bottom_right = np.max(warped_corners, axis=0)  # [x_max, y_max]
        board_width = board_bottom_right[0] - board_top_left[0]
        board_height = board_bottom_right[1] - board_top_left[1]

        # Compute the click's position relative to the board's top-left.
        relative_x = click_point[0] - board_top_left[0]
        relative_y = click_point[1] - board_top_left[1]

        # If the click is outside the board area in warp_and_adjust, ignore it.
        if relative_x < 0 or relative_y < 0 or relative_x > board_width or relative_y > board_height:
            return None

        # The board was warped to an image of size (width_out, height_out) before being resized.
        width_out, height_out = self.original_size_wc

        # Now, warp_and_crop_board resized the board to 640x640.
        # Thus, the scaling factors are:
        scale_x = 640.0 / width_out
        scale_y = 640.0 / height_out

        # Map the relative click position to the 640x640 space.
        mapped_x = relative_x * scale_x
        mapped_y = relative_y * scale_y

        # Optionally, you might check if the mapped point falls within [0, 640] in both directions.
        if mapped_x < 0 or mapped_x > 640 or mapped_y < 0 or mapped_y > 640:
            return None

        return (mapped_x, mapped_y)


"""
# Example Usage
marker_handler = ArMarkerHandler()
marker_handler.create_board()

# Detect corners first
detected_corners = marker_handler.detect_corners()

# Warp and crop using the detected corners
if detected_corners is not None:
    marker_handler.warp_and_crop_board(corners=detected_corners)"""