import os
import time

import cv2
import numpy as np
import pygame


def crop_in_center(frame_bgr):
    # Get the dimensions of the frame
    height, width, _ = frame_bgr.shape

    # Determine the side length of the square
    side_length = min(width, height)

    # Calculate the top-left and bottom-right coordinates for the crop
    x_start = (width - side_length) // 2
    x_end = x_start + side_length
    y_start = (height - side_length) // 2
    y_end = y_start + side_length

    frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Crop the frame to a centered square
    return frame_bgr[y_start:y_end, x_start:x_end]

def resize_frame_bgr_to_target(frame_bgr, sam_target=512):
    """
    Resize the BGR frame to a square sam_target x sam_target (512 x 512 by default),
    preserving aspect ratio and center-cropping.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input frame is empty or not valid")

    h, w, _ = frame_bgr.shape
    scale = sam_target / float(min(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_frame_bgr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return new_frame_bgr

def convert_to_surface(frame_bgr):
    """
    Convert an RGB frame to a Pygame surface.
    Pygame expects (width, height), so we rotate axes accordingly.
    """
    new_frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    surface = pygame.surfarray.make_surface(np.rot90(new_frame_bgr))

    return surface

def capture_frames_for_N_seconds(frame_bgr, end_time, class_name, mask=None, margin=0, proceed=True):
    """
    Captures frames and saves them to 'data/<class_name>/' for a given duration. If a mask is provided,
    the function crops the original image to the bounding box of the mask with a margin and saves only the cropped part.

    Args:
        frame_bgr (np.ndarray): Original BGR frame from the camera.
        end_time (float): Timestamp indicating when to stop capturing.
        class_name (str): Name of the class for saving images.
        mask (np.ndarray, optional): Binary mask indicating the region of interest (same size as frame_bgr).
        margin (int, optional): Number of pixels to expand around the bounding box of the mask. Default is 10.
        proceed (bool): Whether to continue capturing. Default is True.

    Returns:
        tuple: (class_name, proceed)
    """
    directory_path = f'data/{class_name}'
    os.makedirs(directory_path, exist_ok=True)

    if end_time > time.time():
        if mask is not None:
            # Ensure mask is binary
            mask = mask.astype(np.uint8)

            # Find bounding box of the mask
            x, y, w, h = cv2.boundingRect(mask)

            # Apply margin and clip to image boundaries
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(frame_bgr.shape[1], x + w + margin)
            y_max = min(frame_bgr.shape[0], y + h + margin)

            # Crop the image
            cropped_bgr = frame_bgr[y_min:y_max, x_min:x_max]

            # Save the cropped image
            filename = f"{directory_path}/cropped_frame_{time.time_ns()}.jpeg"
            cv2.imwrite(filename, cropped_bgr)
            print(f"Captured cropped frame with mask: {filename}")
        else:
            # Save the entire frame if no mask is provided
            filename = f"{directory_path}/frame_{time.time_ns()}.jpeg"
            cv2.imwrite(filename, frame_bgr)
            print(f"Captured full frame: {filename}")

        return class_name, proceed
    else:
        return class_name, False




# EXPERIMENTAL




