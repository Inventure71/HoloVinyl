import os
import random
import cv2
import shutil
import albumentations as A
from utils.find_objects_and_label import process_image_with_label
from utils.image_processing import remove_similar_images

def create_yolo_dataset(
        input_directory,
        output_directory,
        label="Object",
        img_size = 640,
        target_samples=70,
        train_split=0.8,
        val_split=0.1):
    """
    Create a YOLO-formatted dataset from a directory of images.

    Args:
        input_directory: Source directory containing original images
        output_directory: Base directory for the YOLO dataset
        label: Class label for the objects
        target_samples: Desired number of total samples
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
    """
    # Create directory structure
    dataset_dirs = {
        'train': os.path.join(output_directory, 'train'),
        'val': os.path.join(output_directory, 'val'),
        'test': os.path.join(output_directory, 'test')
    }

    for dir_path in dataset_dirs.values():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Remove duplicate images
    removed = remove_similar_images(input_directory)
    print(f"Removed {len(removed)} similar images")

    # Get list of all images
    original_images = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    n_original = len(original_images)

    if n_original < 1:
        raise ValueError("No valid images found in input directory")

    # Calculate number of augmentations needed
    n_augmentations = max(0, target_samples - n_original)
    augmentations_per_image = n_augmentations // n_original + 1

    # Process and augment images
    all_image_paths = []

    # Define augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.GaussianBlur(p=0.3),
        A.HueSaturationValue(p=0.5),
    ])

    # Process original images and create augmentations
    for img_name in original_images:
        img_path = os.path.join(input_directory, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # Process original image
        corners, _ = process_image_with_label(img_path, "", label=label)

        # Create YOLO format labels
        height, width = image.shape[:2]
        yolo_label = convert_corners_to_yolo(corners, width, height)

        # Save original image and its label
        all_image_paths.append((img_path, yolo_label))

        # Create augmentations
        for i in range(augmentations_per_image):
            if len(all_image_paths) >= target_samples:
                break

            augmented = transform(image=image)
            aug_image = augmented['image']

            # Save augmented image temporarily
            aug_path = os.path.join(input_directory, f"aug_{len(all_image_paths)}_{img_name}")
            cv2.imwrite(aug_path, aug_image)

            # Process augmented image to get new corners
            aug_corners, _ = process_image_with_label(aug_path, "", label=label)
            aug_height, aug_width = aug_image.shape[:2]
            aug_yolo_label = convert_corners_to_yolo(aug_corners, aug_width, aug_height)

            all_image_paths.append((aug_path, aug_yolo_label))

    # Shuffle and split dataset
    random.shuffle(all_image_paths)
    n_samples = len(all_image_paths)
    train_idx = int(n_samples * train_split)
    val_idx = int(n_samples * (train_split + val_split))

    splits = {
        'train': all_image_paths[:train_idx],
        'val': all_image_paths[train_idx:val_idx],
        'test': all_image_paths[val_idx:]
    }

    # Save splits to respective directories
    for split_name, split_data in splits.items():
        for idx, (img_path, yolo_label) in enumerate(split_data):
            # Copy image
            img_filename = f"{split_name}_{idx}{os.path.splitext(img_path)[1]}"
            dst_img_path = os.path.join(dataset_dirs[split_name], 'images', img_filename)
            shutil.copy2(img_path, dst_img_path)

            # Save label
            label_filename = f"{split_name}_{idx}.txt"
            label_path = os.path.join(dataset_dirs[split_name], 'labels', label_filename)
            with open(label_path, 'w') as f:
                f.write(f"0 {' '.join(map(str, yolo_label))}\n")

    # Clean up temporary augmented images
    for img_path, _ in all_image_paths:
        if 'aug_' in img_path:
            os.remove(img_path)

    # Create dataset.yaml
    create_dataset_yaml(output_directory, label)

    print(f"Dataset created successfully with {len(all_image_paths)} images")
    print(f"Train: {len(splits['train'])} images")
    print(f"Validation: {len(splits['val'])} images")
    print(f"Test: {len(splits['test'])} images")


def convert_corners_to_yolo(corners, width, height):
    """Convert corner coordinates to YOLO format (x_center, y_center, w, h)"""
    x_min = min(corner[0] for corner in corners)
    y_min = min(corner[1] for corner in corners)
    x_max = max(corner[0] for corner in corners)
    y_max = max(corner[1] for corner in corners)

    # Convert to YOLO format (normalized)
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height

    return [x_center, y_center, w, h]


def create_dataset_yaml(output_directory, label):
    """Create YAML file for YOLO training"""
    yaml_content = f"""
path: {os.path.abspath(output_directory)}
train: train/images
val: val/images
test: test/images

names:
  0: {label}
    """

    with open(os.path.join(output_directory, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content.strip())


# Example usage
if __name__ == "__main__":
    create_yolo_dataset(
        input_directory="custom_models/raw_images",
        output_directory="custom_models/yolo_dataset",
        label="Object",
        target_samples=70
    )