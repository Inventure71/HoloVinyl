import os
import random
import cv2
import shutil
import albumentations as A
import yaml
from utils.find_objects_and_label import process_image_with_label_V3
from utils.generic import convert_pascal_voc_to_yolo, draw_bounding_box
from utils.image_processing import remove_similar_images
from typing import Dict, List, Tuple, Union, Optional


class YOLODataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.yaml_path = os.path.join(dataset_path, 'dataset.yaml')
        self.config = self.load_yaml_config()

    def load_yaml_config(self) -> dict:
        """Load existing YOLO dataset configuration"""
        if not os.path.exists(self.yaml_path):
            return {
                'path': os.path.abspath(self.dataset_path),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'names': {}
            }

        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml_config(self):
        """Save updated YOLO dataset configuration"""
        with open(self.yaml_path, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)

    def get_next_class_id(self) -> int:
        """Get the next available class ID"""
        existing_ids = [int(k) for k in self.config['names'].keys()]
        return max(existing_ids + [-1]) + 1

    def get_class_counts(self) -> Dict[str, int]:
        """Get the number of images per class in the dataset"""
        class_counts = {name: 0 for name in self.config['names'].values()}

        for split in ['train', 'val', 'test']:
            labels_dir = os.path.join(self.dataset_path, split, 'labels')
            if not os.path.exists(labels_dir):
                continue

            for label_file in os.listdir(labels_dir):
                if not label_file.endswith('.txt'):
                    continue

                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_name = self.config['names'][str(class_id)]
                        class_counts[class_name] += 1

        return class_counts


def create_or_update_yolo_dataset(
        class_directories: Dict[str, str],
        output_directory: str,
        target_samples_per_class: int = 70,
        train_split: float = 0.8,
        val_split: float = 0.1,
        debug_boundaries: bool = False,
        existing_dataset: Optional[str] = None
):
    """
    Create a new YOLO dataset or update an existing one with new classes.

    Args:
        class_directories: Dictionary mapping class names to their input directories
        output_directory: Base directory for the YOLO dataset
        target_samples_per_class: Desired number of samples per class
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        existing_dataset: Path to existing dataset to update (optional)
    """
    # Initialize or load existing dataset
    if existing_dataset:
        print(f"Loading existing dataset from {existing_dataset}")
        dataset = YOLODataset(existing_dataset)
        output_directory = existing_dataset
    else:
        print("Creating new dataset")
        os.makedirs(output_directory, exist_ok=True)
        dataset = YOLODataset(output_directory)

    # Create directory structure if needed
    dataset_dirs = {
        'train': os.path.join(output_directory, 'train'),
        'val': os.path.join(output_directory, 'val'),
        'test': os.path.join(output_directory, 'test')
    }

    for dir_path in dataset_dirs.values():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # NOTE: DEBUG ONLY CAN BE REMOVED:
    if debug_boundaries:
        # **Define Debug Directory Structure**
        debug_output_directory = "custom_models/debug_images"  # You can change this path as needed
        debug_dataset_dirs = {
            'train': os.path.join(debug_output_directory, 'train'),
            'val': os.path.join(debug_output_directory, 'val'),
            'test': os.path.join(debug_output_directory, 'test')
        }

        for dir_path in debug_dataset_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    # Process new classes
    all_image_paths = []  # List of (image_path, label_data, class_id, class_name)

    for class_name, input_directory in class_directories.items():
        # Skip if class already exists
        if class_name in dataset.config['names'].values():
            print(f"Class {class_name} already exists in dataset, skipping...")
            continue

        class_id = dataset.get_next_class_id()
        print(f"Processing new class: {class_name} (ID: {class_id})")

        # Add class to configuration
        dataset.config['names'][str(class_id)] = class_name

        # Process images for this class
        class_image_paths = process_class_images(
            input_directory,
            class_name,
            class_id,
            target_samples_per_class
        )
        # Include class_name in the tuple for debugging
        class_image_paths = [
            (img_path, yolo_label, class_id, class_name) for (img_path, yolo_label, class_id) in class_image_paths
        ]
        all_image_paths.extend(class_image_paths)

    if not all_image_paths:
        print("No new classes to process")
        return

    # Split new data
    random.shuffle(all_image_paths)
    n_samples = len(all_image_paths)
    train_idx = int(n_samples * train_split)
    val_idx = int(n_samples * (train_split + val_split))

    splits = {
        'train': all_image_paths[:train_idx],
        'val': all_image_paths[train_idx:val_idx],
        'test': all_image_paths[val_idx:]
    }

    # Get current maximum index for each split
    current_indices = {}
    for split_name in splits.keys():
        images_dir = os.path.join(dataset_dirs[split_name], 'images')
        existing_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        current_indices[split_name] = len(existing_files)

    # Save new data to respective directories and save debug copies
    for split_name, split_data in splits.items():
        start_idx = current_indices[split_name]
        for idx, (img_path, yolo_label, class_id, class_name) in enumerate(split_data, start=start_idx):
            # Copy image
            img_ext = os.path.splitext(img_path)[1]
            img_filename = f"{split_name}_{idx}{img_ext}"
            dst_img_path = os.path.join(dataset_dirs[split_name], 'images', img_filename)
            shutil.copy2(img_path, dst_img_path)

            # Save label
            label_filename = f"{split_name}_{idx}.txt"
            label_path = os.path.join(dataset_dirs[split_name], 'labels', label_filename)
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {' '.join(map(str, yolo_label))}\n")

            # NOTE: DEBUG ONLY CAN BE REMOVED:
            if debug_boundaries:
                # **Save Debug Image with Bounding Box**
                debug_img_path = os.path.join(debug_dataset_dirs[split_name], img_filename)
                draw_bounding_box(
                    image_path=dst_img_path,
                    yolo_bbox=yolo_label,
                    class_id=class_id,
                    class_name=class_name,
                    output_path=debug_img_path
                )

    # Clean up temporary augmented images
    for img_path, _, _, _ in all_image_paths:
        if 'aug_' in img_path:
            os.remove(img_path)

    # Save updated configuration
    dataset.save_yaml_config()

    # Print statistics
    print("\nDataset update completed:")
    class_counts = dataset.get_class_counts()
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")


def process_class_images(
    input_directory: str,
    class_name: str,
    class_id: int,
    target_samples: int
) -> List[Tuple[str, List[float], int]]:
    """Process images for a single class, including augmentation if needed."""
    # Remove duplicate images
    removed = remove_similar_images(input_directory)
    print(f"Removed {len(removed)} similar images from {class_name}")

    # Get list of all images
    original_images = [
        f for f in os.listdir(input_directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    n_original = len(original_images)
    if n_original < 1:
        raise ValueError(f"No valid images found in input directory for class {class_name}")

    # Calculate how many augmentations are needed
    n_augmentations = max(0, target_samples - n_original)
    augmentations_per_image = n_augmentations // n_original + 1

    # Define Albumentations transform with bbox_params
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
            A.GaussianBlur(p=0.3),
            A.HueSaturationValue(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3,
        )
    )

    class_image_paths = []  # Will store tuples of (img_path, yolo_label, class_id)

    for img_name in original_images:
        img_path = os.path.join(input_directory, img_name)
        image_cv = cv2.imread(img_path)

        if image_cv is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        # 1) Get bounding box once (in x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = process_image_with_label_V3(img_path, label=class_name)
        # Make sure the box is within the image boundary
        h, w = image_cv.shape[:2]
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(w, x_max); y_max = min(h, y_max)

        # 2) Append the original image (no augmentation)
        # Convert corners to YOLO for the label
        yolo_label = convert_pascal_voc_to_yolo(x_min, y_min, x_max, y_max, w, h)
        class_image_paths.append((img_path, yolo_label, class_id))

        # 3) Create augmentations if needed
        for i in range(augmentations_per_image):
            if len(class_image_paths) >= target_samples:
                break

            # Albumentations expects the bounding box in (x_min, y_min, x_max, y_max)
            # and a list of labels that correspond to each box
            augmented = transform(
                image=image_cv,
                bboxes=[(x_min, y_min, x_max, y_max)],
                class_labels=[class_name]  # label_fields
            )
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']  # List of transformed bboxes

            if not aug_bboxes:
                # In some transforms, the box can disappear (min_visibility)
                continue

            # We only have one box
            aug_xmin, aug_ymin, aug_xmax, aug_ymax = aug_bboxes[0]

            # Write augmented image to a file
            aug_path = os.path.join(input_directory, f"aug_{len(class_image_paths)}_{img_name}")
            cv2.imwrite(aug_path, aug_image)

            # Convert the bounding box to YOLO
            aug_h, aug_w = aug_image.shape[:2]
            aug_yolo_label = convert_pascal_voc_to_yolo(
                aug_xmin, aug_ymin, aug_xmax, aug_ymax, aug_w, aug_h
            )

            # Add to the list
            class_image_paths.append((aug_path, aug_yolo_label, class_id))

    return class_image_paths


# Example usage
if __name__ == "__main__":
    # Create new dataset
    class_dirs = {
        "car": "car_images_dir",
        "truck": "truck_images_dir"
    }

    create_or_update_yolo_dataset(
        class_directories=class_dirs,
        output_directory="yolo_dataset",
        target_samples_per_class=70
    )

    # Later, add a new class to existing dataset
    new_class_dirs = {
        "motorcycle": "motorcycle_images_dir"
    }

    create_or_update_yolo_dataset(
        class_directories=new_class_dirs,
        output_directory="yolo_dataset",
        target_samples_per_class=70,
        existing_dataset="yolo_dataset"  # Path to existing dataset
    )