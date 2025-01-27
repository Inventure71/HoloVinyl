import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Handler:
    def __init__(self, checkpoint, model_cfg, device=None):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg

        # Initialize device once
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print("\nWarning: MPS device support is preliminary.")

        # Load SAM2 model once
        self.sam2 = build_sam2(
            self.model_cfg,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=True
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2, nms_threshold=0.1) #mask_resolution=128, sampling_density=0.1
        self.predictor = SAM2ImagePredictor(self.sam2)

    def generate_masks(self, image_rgb):
        """
        Generate masks for an RGB image using the SAM2 AutomaticMaskGenerator.
        """
        return self.mask_generator.generate(image_rgb)

    @staticmethod
    def preprocess_image(image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path)
        return np.array(image.convert("RGB"))

    def get_sam_masks_and_boxes(self, frame_bgr):
        """
        Given a BGR frame, prepare it for SAM (512x512),
        generate masks, then return (sam_input_bgr, [mask_annots], [boxes]).
        """
        # 1) Prepare the frame for SAM
        #sam_input_bgr = prepare_frame_for_sam(frame_bgr, sam_target=256) # 512

        # 2) Convert to RGB for SAM
        #sam_input_rgb = cv2.cvtColor(sam_input_bgr, cv2.COLOR_BGR2RGB)

        # 3) Generate masks
        masks = self.generate_masks(frame_bgr)

        # 4) Extract bounding boxes
        boxes = self.extract_bounding_boxes(masks)
        return masks, boxes

    #PROBABLY NOT USED
    @staticmethod
    def extract_bounding_boxes(anns):
        """
        Convert the 'segmentation' in each annotation to bounding boxes (rotated),
        returned as 4 corner points.
        """
        bounding_boxes = []
        for ann in anns:
            seg = ann["segmentation"]
            contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                bounding_boxes.append(box)
        return bounding_boxes

    # PROBABLY NOT USED
    def display_masks_with_boxes(self, image, image_path=None):
        """Generate and display masks with bounding boxes for a given image"""
        #image = self.preprocess_image(frame)
        masks = self.generate_masks(image)
        print(f"Number of masks: {len(masks)}")
        print(f"Mask keys: {masks[0].keys()}")

        bounding_boxes = self.extract_bounding_boxes(masks)

        #plt.figure(figsize=(20, 20))
        #plt.imshow(image)
        #self.show_anns(image, masks)

        # Draw bounding boxes
        for box in bounding_boxes:
            plt.plot(
                [box[0][0], box[1][0], box[2][0], box[3][0], box[0][0]],
                [box[0][1], box[1][1], box[2][1], box[3][1], box[0][1]],
                color="red",
                linewidth=2,
            )

        #plt.axis('off')
        #plt.show()

        # Return bounding boxes relative to the original image
        return image, masks, bounding_boxes

    def generate_mask_with_point(self, frame_bgr, point=None, point_label=1):
        """
        Generate a single mask that includes the specified point.

        Args:
            image_rgb (np.ndarray): The input RGB image.
            point (tuple): The point to include in the mask, given as (x, y).
            point_label (int): Label of the point (1 = foreground, 0 = background).

        Returns:
            dict: The selected mask annotation.
        """
        sam_target, _, _ = frame_bgr.shape
        point = (sam_target//2, sam_target//2) if point is None else point

        self.predictor.set_image(frame_bgr)

        print(f"Point: {point}")

        # Pass the point to SAM2's predictor
        input_point = np.array([point])
        input_label = np.array([point_label])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  # Generate only one mask
        )

        # Select the single mask (since multimask_output=False)
        mask = masks[0]
        return {
            "segmentation": mask,
            "score": scores[0]
        }

    def generate_mask_with_square(self, frame_bgr, square_center=None, square_size=40, point_label=1):
        """
        Generate a single mask that includes a small square region.

        Args:
            image (np.ndarray): The input RGB image.
            square_center (tuple): The (x, y) center of the square in SAM's coordinate space.
            square_size (int): The side length of the square.
            point_label (int): Label of the points in the square (1 = foreground, 0 = background).

        Returns:
            dict: The selected mask annotation.
        """
        h, w, _ = frame_bgr.shape
        sam_target = h
        if square_center is None:
            square_center = (sam_target // 2, sam_target // 2)

        # Calculate square bounds in SAM's coordinate space
        x_center, y_center = square_center
        half_size = square_size // 2
        square_points = [
            (x_center - half_size, y_center - half_size),
            (x_center + half_size, y_center - half_size),
            (x_center - half_size, y_center + half_size),
            (x_center + half_size, y_center + half_size),
        ]


        # Set the image for the SAM predictor
        self.predictor.set_image(frame_bgr)

        # Pass the square points to SAM's predictor
        input_points = np.array(square_points)
        input_labels = np.full(len(square_points), point_label)  # Use the same label for all points

        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False  # Generate only one mask
        )

        # Select the single mask (since multimask_output=False)
        mask = masks[0]
        return {
            "segmentation": mask,
            "score": scores[0],
            "square_bounds": (x_center - half_size, y_center - half_size, x_center + half_size, y_center + half_size),
        }


