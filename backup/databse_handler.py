import os
import random

import cv2
import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np

from test import process_image_with_label
from utils.image_processing import remove_similar_images


def crete_dataset_from_directory(directory, output_file, label="Object", target_samples=70):
    #os.makedirs(output_file, exist_ok=True)

    # remove duplicates
    removed = remove_similar_images(directory)
    print("Removed images:", removed)

    # count images in directory
    n_images = len(os.listdir(directory))

    for extra_image in range(0, n_images-target_samples):
        # get random picture in directory
        random_image = random.choice(os.listdir(directory))
        augment_image(random_image, directory, n_samples=target_samples)

    for image in os.listdir(directory):
        label_corners, _ = process_image_with_label(image, "", label=label)

def augment_image(image_path, output_dir, n_samples=70):
    # Read the image
    image = cv2.imread(image_path)

    # Define augmentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.GaussianBlur(p=0.3),
        A.HueSaturationValue(p=0.5),
        A.RandomCrop(height=image.shape[0] - 50, width=image.shape[1] - 50, p=0.5),
    ])

    # Apply augmentations and save images
    for i in range(n_samples):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        output_path = f"{output_dir}/augmented_{i}.png"
        cv2.imwrite(output_path, augmented_image)
        print(f"Saved: {output_path}")


# Example usage
image_path = "img.png"
output_dir = "augmented_images"
augment_image(image_path, output_dir, n_samples=10)
