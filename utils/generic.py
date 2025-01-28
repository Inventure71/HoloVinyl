import json
import os

import cv2


def load_mappings(mapping_file_path="variables/class_mappings.json"):
    if os.path.exists(mapping_file_path):
        with open(mapping_file_path, "r") as f:
            return json.load(f)
    return {}

def save_mappings(mappings, mapping_file_path="variables/class_mappings.json"):
    with open(mapping_file_path, "w") as f:
        json.dump(mappings, f, indent=4)

def convert_corners_to_yolo(corners, width, height):
    """Convert corner coordinates to YOLO format (x_center, y_center, w, h)"""
    x_min = min(corner[0] for corner in corners)
    y_min = min(corner[1] for corner in corners)
    x_max = max(corner[0] for corner in corners)
    y_max = max(corner[1] for corner in corners)

    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height

    return [x_center, y_center, w, h]

def convert_pascal_voc_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Converts bounding box from PASCAL VOC to YOLO format.
    x_min, y_min, x_max, y_max are absolute pixel coordinates.
    """
    # Clip values in case of any rounding issues
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # Center x, y
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    # Width and height
    w = x_max - x_min
    h = y_max - y_min

    # Normalize
    x_center /= float(img_width)
    y_center /= float(img_height)
    w /= float(img_width)
    h /= float(img_height)

    return [x_center, y_center, w, h]

def convert_yolo_to_pixel(yolo_bbox, img_width, img_height):
    """
    Converts YOLO format bounding box to pixel coordinates.
    yolo_bbox: List or tuple with [x_center, y_center, width, height] in normalized format.
    Returns: (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    return (x_min, y_min, x_max, y_max)

def draw_bounding_box(image_path, yolo_bbox, class_id, class_name, output_path):
    """
    Draws a bounding box on the image and saves it to the output path.

    Args:
        image_path (str): Path to the original image.
        yolo_bbox (List[float]): YOLO format bounding box [x_center, y_center, w, h].
        class_id (int): ID of the class.
        class_name (str): Name of the class.
        output_path (str): Path to save the annotated image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path} for drawing bounding box.")
        return

    img_height, img_width = image.shape[:2]

    # Convert YOLO bbox to pixel coordinates
    x_min, y_min, x_max, y_max = convert_yolo_to_pixel(yolo_bbox, img_width, img_height)

    # Draw the bounding box
    color = (0, 255, 0)  # Green color for bounding box
    thickness = 2
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # Put class label text above the bounding box
    label = f"{class_name} ({class_id})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    text_origin = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)

    # Draw background rectangle for text for better visibility
    cv2.rectangle(
        image,
        (text_origin[0], text_origin[1] - text_size[1]),
        (text_origin[0] + text_size[0], text_origin[1] + 5),
        color,
        cv2.FILLED
    )
    # Put text on image
    cv2.putText(
        image,
        label,
        text_origin,
        font,
        font_scale,
        (0, 0, 0),  # Black text
        font_thickness,
        cv2.LINE_AA
    )

    # Save the annotated image
    cv2.imwrite(output_path, image)
