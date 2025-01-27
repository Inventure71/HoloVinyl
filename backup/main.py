import os
import cv2
import time
import torch
import numpy as np
import pygame
from pygame.locals import *
import yaml
import shutil

from ultralytics import YOLO

from SAM_handler import SAM2Handler
from image_handling_v2 import resize_frame_bgr_to_target

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
YOLO_MODEL_PATH = "yolo11n.pt"  # Pretrained YOLO weights
CHECKPOINT_PATH = r"C:\Users\matte\PycharmProjects\sam2\checkpoints\sam2.1_hiera_tiny.pt"
MODEL_CFG_PATH = r"C:\Users\matte\PycharmProjects\sam2\sam2\configs\sam2.1\sam2.1_hiera_t.yaml"

NEW_CLASS_NAME = "new_object"  # Name of the new class you want to add
DATASET_PATH = "datasets/new_class"
FPS = 20  # Camera capture FPS
CAPTURE_SECONDS = 5  # Capture duration in seconds

CLASSES_YAML_PATH = "classes.yaml"  # Path to store class names
RETRAINED_WEIGHTS_PATH = "retrain_runs/exp/weights/best.pt"  # Path to retrained weights

# Create the dataset path if it doesn't exist
os.makedirs(DATASET_PATH, exist_ok=True)
images_path = os.path.join(DATASET_PATH, "images")
labels_path = os.path.join(DATASET_PATH, "labels")
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def load_class_names(yolo_model_path, retrained_weights_path):
    """
    Load existing class names from the YOLO model or classes.yaml.
    Append the new class name if it's not already present.
    Save the updated class names to a YAML file.
    """
    if os.path.exists(retrained_weights_path):
        print(f"Loading retrained model from: {retrained_weights_path}")
        model = YOLO(retrained_weights_path)
    else:
        print(f"Loading pre-trained model from: {yolo_model_path}")
        model = YOLO(yolo_model_path)

    # Debug: Print type of model.names
    print(f"Type of model.names: {type(model.names)}")

    # Check the type of model.names
    if isinstance(model.names, dict):
        # Convert dict to list
        existing_names = [model.names[i] for i in sorted(model.names.keys())]
    elif isinstance(model.names, list):
        existing_names = model.names.copy()
    else:
        raise TypeError(f"model.names is of unsupported type: {type(model.names)}")

    # Debug: Print existing_names
    print(f"Existing class names ({len(existing_names)}): {existing_names}")

    # If classes.yaml exists, load it instead to maintain consistency
    if os.path.exists(CLASSES_YAML_PATH):
        print(f"Loading class names from {CLASSES_YAML_PATH}")
        with open(CLASSES_YAML_PATH, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data and isinstance(data['names'], list):
                existing_names = data['names']
                print(f"Loaded class names from {CLASSES_YAML_PATH}: {existing_names}")
            else:
                print(
                    f"Warning: 'names' key missing or not a list in {CLASSES_YAML_PATH}. Using existing class names from model.")
    else:
        # Initialize classes.yaml with existing class names
        with open(CLASSES_YAML_PATH, 'w') as f:
            yaml.dump({'names': existing_names}, f)
        print(f"Created {CLASSES_YAML_PATH} with existing class names.")

    # Check if the new class already exists
    if NEW_CLASS_NAME not in existing_names:
        existing_names.append(NEW_CLASS_NAME)
        # Save updated class names
        with open(CLASSES_YAML_PATH, 'w') as f:
            yaml.dump({'names': existing_names}, f)
        print(f"Added '{NEW_CLASS_NAME}' to {CLASSES_YAML_PATH}")
    else:
        print(f"'{NEW_CLASS_NAME}' already exists in class names.")

    # Determine the NEW_CLASS_ID based on the updated class list
    try:
        NEW_CLASS_ID = existing_names.index(NEW_CLASS_NAME)
        print(f"Assigned NEW_CLASS_ID: {NEW_CLASS_ID} for class '{NEW_CLASS_NAME}'")
    except ValueError:
        print(f"Error: '{NEW_CLASS_NAME}' not found in class names.")
        NEW_CLASS_ID = None

    return model, existing_names, NEW_CLASS_ID


def save_classes_yaml(class_names):
    """
    Save class names to classes.yaml
    """
    with open(CLASSES_YAML_PATH, 'w') as f:
        yaml.dump({'names': class_names}, f)
    print(f"Updated {CLASSES_YAML_PATH} with class names.")


# ---------------------------------------------------
# SAM2 Masking
# ---------------------------------------------------
def get_sam2_mask(frame_bgr):
    """
    Uses SAM2 with a single “inclusion point” at the image center.
    Returns a boolean mask (H x W), where True = object, False = background.
    """
    # Resize frame for SAM2 if needed
    h, w, _ = frame_bgr.shape
    resized_frame = resize_frame_bgr_to_target(frame_bgr, 512)

    # Coordinates in the resized image’s coordinate system
    # We'll pick the center of the resized image as our single inclusion point
    resized_h, resized_w, _ = resized_frame.shape
    center_point = (resized_w // 2, resized_h // 2)

    # Generate mask with a single inclusion point
    result_dict = sam2_handler.generate_mask_with_point(
        frame_bgr=resized_frame,
        point=center_point,
        point_label=1
    )

    # The returned mask is in the shape of the resized frame. We need to
    # map it back to the original frame size if we want the bounding box in original coords.
    mask_resized = result_dict.get("segmentation", None)
    if mask_resized is None:
        print("Warning: SAM2 did not return a 'segmentation' mask.")
        return np.zeros((h, w), dtype=bool)

    # Resize the mask back to the original frame size
    mask_original = cv2.resize(
        mask_resized.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    return mask_original


# ---------------------------------------------------
# Prediction Drawing
# ---------------------------------------------------
def draw_yolo_predictions(surface, predictions, class_names):
    """
    Draw bounding boxes from YOLO predictions on a Pygame surface.
    predictions: Ultralytics YOLO v8 output (list of boxes, etc.)
    """
    for box in predictions.boxes:
        # Extract box coordinates and other attributes
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {conf:.2f}"
        else:
            label = f"ID {class_id}: {conf:.2f}"  # Fallback if class_id is out of range

        # Draw rectangle
        pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)

        # Render label
        font = pygame.font.SysFont("Arial", 16)
        text_surface = font.render(label, True, (0, 255, 0))
        surface.blit(text_surface, (x1, max(0, y1 - 20)))


# ---------------------------------------------------
# Bounding Box from Mask
# ---------------------------------------------------
def bounding_box_from_mask(binary_mask):
    """
    Convert a binary mask to a bounding box [x_min, y_min, x_max, y_max].
    Returns None if no foreground pixels are found.
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None  # No foreground found
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]


# ---------------------------------------------------
# Image Capture
# ---------------------------------------------------
def capture_images(cap):
    """
    Capture images for CAPTURE_SECONDS and save them to images_path.
    Blocks for the duration of capture.
    """
    print(">>> Starting image capture.")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera.")
            break

        # Save image
        img_filename = f"frame_{int(time.time())}_{frame_count}.jpg"
        img_path = os.path.join(images_path, img_filename)
        cv2.imwrite(img_path, frame)
        print(f"Captured {img_path}")
        frame_count += 1

    print(">>> Image capture completed.")


# ---------------------------------------------------
# Image Processing
# ---------------------------------------------------
def process_images(class_names, new_class_id):
    """
    Process each captured image through SAM2, create bounding boxes, and save labels.
    """
    print(">>> Starting image processing with SAM2.")
    image_files = [f for f in os.listdir(images_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to read {img_path}. Skipping.")
            continue

        # Get SAM2 mask
        mask = get_sam2_mask(frame)  # shape (H, W), bool

        # Convert mask to bounding box
        bbox = bounding_box_from_mask(mask)
        if bbox:
            x_min, y_min, x_max, y_max = bbox

            height, width, _ = frame.shape

            # YOLO label format = class_id x_center y_center width height (normalized)
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            w = x_max - x_min
            h = y_max - y_min

            # Normalize
            x_center_norm = x_center / width
            y_center_norm = y_center / height
            w_norm = w / width
            h_norm = h / height

            # Save label
            label_filename = f"{os.path.splitext(img_file)[0]}.txt"
            label_path = os.path.join(labels_path, label_filename)

            # IMPORTANT: YOLO expects integer class IDs. Use new_class_id
            with open(label_path, "w") as f:
                f.write(f"{new_class_id} {x_center_norm:.6f} {y_center_norm:.6f} "
                        f"{w_norm:.6f} {h_norm:.6f}\n")

            print(f"Processed {img_file} -> {label_filename}")
        else:
            print(f"No mask found for {img_file}. Skipping label creation.")

    print(">>> Image processing completed.")


# ---------------------------------------------------
# YOLO Retraining
# ---------------------------------------------------
def retrain_yolo(yolo_model_path, retrained_weights_path, class_names):
    """
    Retrain YOLO using the newly collected data.
    Uses absolute paths in data_temp.yaml to avoid path issues.
    """
    print(">>> Starting YOLO retraining (this might take a while).")

    # Create a unique project name based on current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    project_name = f"retrain_runs/exp_{timestamp}"

    # Create a minimal data.yaml with absolute paths
    data_yaml_path = "data_temp.yaml"
    train_images_abs_path = os.path.abspath(images_path)
    val_images_abs_path = os.path.abspath(images_path)  # Using the same for validation in this example

    data_yaml = {
        'train': train_images_abs_path,
        'val': val_images_abs_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"Created {data_yaml_path} with:")
    print(f"  train: {train_images_abs_path}")
    print(f"  val: {val_images_abs_path}")
    print(f"  nc: {len(class_names)}")
    print(f"  names: {class_names}")

    # Train the model
    new_model = YOLO(yolo_model_path)
    new_model.train(
        data=data_yaml_path,
        epochs=5,  # Minimal number of epochs for demonstration
        imgsz=640,
        project=project_name,
        name="exp",
        exist_ok=True
    )

    # Typically, best weights are saved as:
    best_weights_path = os.path.join(project_name, "exp", "weights", "best.pt")
    if os.path.exists(best_weights_path):
        print(f"Training complete. Saving retrained weights to {retrained_weights_path}")
        # Ensure the retrain_runs directory exists
        os.makedirs(os.path.dirname(retrained_weights_path), exist_ok=True)
        # Copy the best weights to a fixed retrained_weights_path
        shutil.copy(best_weights_path, retrained_weights_path)
        print(f"Copied best weights to {retrained_weights_path}")
        return retrained_weights_path
    else:
        print("Warning: best weights not found, check your training directory.")
        return None


# ---------------------------------------------------
# PyGame Application
# ---------------------------------------------------
def main():
    global model

    # Load class names and assign NEW_CLASS_ID
    model, class_names, new_class_id = load_class_names(YOLO_MODEL_PATH, RETRAINED_WEIGHTS_PATH)

    # Check if NEW_CLASS_ID was assigned
    if new_class_id is None:
        print("Error: NEW_CLASS_ID not assigned. Exiting.")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Your camera source; adjust index if needed
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Initialize PyGame
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)

    # Determine a suitable window size (e.g., 640x480)
    window_width = 640
    window_height = 480
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("YOLO + SAM2 Integration")

    clock = pygame.time.Clock()

    show_detections = True
    running = True

    print("Instructions:")
    print(" - Press 'D' to toggle YOLO detection overlay.")
    print(" - Press 'A' to add new class data (capture images, process with SAM2, retrain).")
    print(" - Press 'Q' or close window to quit.")

    while running:
        # --- Handle Events ---
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    running = False
                elif event.key == K_d:
                    show_detections = not show_detections
                    print(f"[INFO] Detection overlay {'ON' if show_detections else 'OFF'}.")
                elif event.key == K_a:
                    # 1) Capture images
                    capture_images(cap)
                    # 2) Process images with SAM2 to create labels
                    process_images(class_names, new_class_id)
                    # 3) Retrain YOLO
                    best_weights = retrain_yolo(YOLO_MODEL_PATH, RETRAINED_WEIGHTS_PATH, class_names)
                    # 4) Reload the YOLO model
                    if best_weights:
                        model = YOLO(best_weights)
                        print("YOLO model reloaded with new weights.")
                        # Reload class names to ensure consistency
                        model, class_names, new_class_id = load_class_names(YOLO_MODEL_PATH, best_weights)
                        print("Class names reloaded.")

        # --- Read Frame from Camera ---
        ret, frame_bgr = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Resize for display (optional)
        display_frame = cv2.resize(frame_bgr, (window_width, window_height))
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # YOLO inference
        if show_detections:
            results = model(frame_rgb, verbose=False)
            # Uncomment the following line to debug predictions
            # print("Predictions:", results[0].boxes)  # Check if boxes are detected

            # Convert to Pygame surface
            surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            draw_yolo_predictions(surface, results[0], class_names)
        else:
            # No inference, just convert to surface
            surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

        # --- Draw UI ---
        screen.blit(surface, (0, 0))

        # Display instructions on screen
        instructions = [
            "Press 'D' to toggle detection overlay",
            "Press 'A' to capture images, label with SAM2, and retrain YOLO",
            "Press 'Q' or close window to quit",
        ]
        y_offset = 5
        for text in instructions:
            text_surface = font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (5, y_offset))
            y_offset += 20

        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    cap.release()
    pygame.quit()
    print(">>> Application closed.")


if __name__ == "__main__":
    # Initialize SAM2 Handler
    sam2_handler = SAM2Handler(checkpoint=CHECKPOINT_PATH, model_cfg=MODEL_CFG_PATH)
    main()
