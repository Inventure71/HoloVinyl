import cv2
import numpy as np
import rembg
from PIL import Image
from PIL import ImageDraw


def process_image_with_label_V3(image_path, label="Object"):
    """
    Returns bounding box as (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
    """
    # Load the input image
    input_image = Image.open(image_path).convert("RGBA")

    # Remove background using rembg
    output_array = rembg.remove(np.array(input_image))

    # Create a binary mask from the alpha channel
    alpha_mask = output_array[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object detected in the image")

    # Calculate image center
    image_center = (binary_mask.shape[1] // 2, binary_mask.shape[0] // 2)

    # Function to compute distance from contour centroid to image center
    def contour_distance(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return float('inf')
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return np.sqrt((cX - image_center[0])**2 + (cY - image_center[1])**2)

    # Find contour closest to center
    closest_contour = min(contours, key=contour_distance)

    # Create a mask for the closest contour
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, [closest_contour], -1, 255, thickness=cv2.FILLED)

    # Add margin by dilating the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # Adjust the kernel size as needed
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find bounding rectangle of the cleaned mask
    x, y, w, h = cv2.boundingRect(mask)

    # Return box in (x_min, y_min, x_max, y_max)
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    return (x_min, y_min, x_max, y_max)


def process_image_with_label_V2(image_path, label="Object"):
    """
    Returns bounding box as (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
    """
    # Load the input image
    input_image = Image.open(image_path).convert("RGBA")

    # Remove background using rembg
    output_array = rembg.remove(np.array(input_image))

    # Create a binary mask from the alpha channel
    alpha_mask = output_array[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object detected in the image")

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the min area rectangle or bounding rect
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Return box in (x_min, y_min, x_max, y_max)
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    return (x_min, y_min, x_max, y_max)


def process_image_with_label_old(image_path, output_path, label="Object", save_image=False):
    # Load the input image
    input_image = Image.open(image_path).convert("RGBA")

    # Remove background using rembg
    output_array = rembg.remove(np.array(input_image))

    # Create a binary mask from the alpha channel
    alpha_mask = output_array[:, :, 3]

    # Convert mask to binary (0 or 255)
    _, binary_mask = cv2.threshold(alpha_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    if not contours:
        raise ValueError("No object detected in the image")

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int_)

    # Sort points to get consistent order: top-left, top-right, bottom-right, bottom-left
    # First sort by y-coordinate (top to bottom)
    box = box[np.argsort(box[:, 1])]

    # Sort top two points by x coordinate (left to right)
    box[:2] = box[:2][np.argsort(box[:2, 0])]

    # Sort bottom two points by x coordinate (left to right)
    box[2:] = box[2:][np.argsort(box[2:, 0])]
    # invert bottom two points
    box[2:] = box[2:][::-1]

    # Create annotated image
    annotated_image = input_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Draw bounding box
    points = [(int(x), int(y)) for x, y in box]
    draw.line(points + [points[0]], fill="red", width=3)

    # Add label above the top-left corner
    text_position = (points[0][0], points[0][1] - 20)
    draw.text(text_position, label, fill="red")

    #if save_image:
        # Save the annotated image
    #    annotated_image.save(output_path)

    # Convert box points to list of tuples for easier handling
    box_points = [(int(x), int(y)) for x, y in box]

    return box_points, annotated_image


# Example usage
if __name__ == "__main__":
    image_path = "img_1.png"
    output_path = "output_image_with_label.png"
    label = "Primary Object"

    try:
        bounding_box, annotated_image = process_image_with_label_old(image_path, output_path, label)
        print("Bounding Box Coordinates (clockwise from top-left):", bounding_box)
    except Exception as e:
        print(f"Error processing image: {str(e)}")