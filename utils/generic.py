import json
import os


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
