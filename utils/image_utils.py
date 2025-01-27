import os

import cv2
import numpy as np
from PIL import Image
import imagehash

def capture_image_from_frame(frame, output_path, img_size = 640):
    """
    Capture an image from a video frame and save it to the output path.
    :param frame: Video frame to capture the image from.
    :param output_path: Path to save the captured image.
    """
    image = Image.fromarray(frame)
    image = image.resize((img_size,img_size))
    image.save(output_path)

# Function to remove identical or too similar images
def remove_similar_images(image_dir, threshold=5):
    # Load all image paths
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith((".png", ".jpg", ".jpeg"))]

    # Dictionary to store hash values
    hashes = {}
    removed_images = []

    for image_path in image_paths:
        image = Image.open(image_path)
        hash_value = imagehash.phash(image)  # Compute perceptual hash

        # Check for similar images
        duplicate_found = False
        for existing_hash in hashes:
            if abs(hash_value - existing_hash) <= threshold:  # Compare hash values
                duplicate_found = True
                removed_images.append(image_path)
                os.remove(image_path)  # Remove similar image
                break

        if not duplicate_found:
            hashes[hash_value] = image_path

    print(f"Removed {len(removed_images)} similar images.")
    return removed_images

def transform_to_square(frame, points):
    """
    Applies perspective transformation to warp the board to a square view.
    """
    if len(points) != 4:
        print("Error: Calibration not complete. Can't transform.")
        return frame

    # Define the desired square view
    side_length = 500  # Define the output size of the square
    dst = np.array([
        [0, 0],
        [side_length - 1, 0],
        [side_length - 1, side_length - 1],
        [0, side_length - 1]
    ], dtype="float32")

    # Apply perspective transformation
    src = np.array(points, dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, matrix, (side_length, side_length))
    return warped
