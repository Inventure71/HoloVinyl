import torch
from ultralytics import YOLO
from typing import List, Union

from find_parameters_for_training import find_parameters


class YOLOHandler:
    def __init__(self, model_path: str = None):
        """
        Initialize the YOLO handler.
        :param model_path: Path to a pretrained or retrained YOLO model.
                           If None, you'll need to load or train a model later.
        """

        self.params = find_parameters()
        self.device = self.params["device"]

        if model_path:
            self.model = YOLO(model_path)  # Load an existing model
            print(f"Model loaded from: {model_path}")
        else:
            self.model = None

    def train_model(self, data_path: str, model_type: str = "yolov11n.pt", epochs: int = 50, batch_size: int = 16,
                    img_size: int = 640, save_dir: str = "custom_models/runs/train"):
        """
        Train a YOLO model on a custom dataset.
        :param data_path: Path to the data.yaml file.
        :param model_type: Pretrained YOLO model to use as a starting point (e.g., yolov8n.pt).
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param img_size: Image size for training.
        :param save_dir: Directory to save training results.
        """
        self.model = YOLO(model_type)  # Load a base YOLO model

        # Use optimized settings
        batch_size = self.params["batch_size"]
        img_size = img_size
        num_workers = self.params["num_workers"]
        cache = self.params["use_cache"]


        self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            save_dir=save_dir,
            cache=cache,  # Dramatically increases speed if set to True
            workers=num_workers,
        )
        """
        self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            device=self.device,
            save_dir=save_dir,
        )"""

        print(f"Training completed. Results saved in: {save_dir}")

    def load_model(self, model_path: str, fallback="yolo11n.pt"):
        """
        Load an already trained YOLO model.
        :param model_path: Path to the model file (e.g., best.pt).
        """
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            self.model = YOLO(fallback)


    def get_classes(self) -> List[str]:
        """
        Retrieve all known classes in the model.
        :return: A list of class names.
        """
        if not self.model:
            raise ValueError("No model loaded. Load or train a model first.")

        # Ensure the model has a class names attribute
        if hasattr(self.model, 'names'):
            return list(self.model.names.values())
        else:
            print("No class names found in the model.")
            return []
            #raise AttributeError("The loaded model does not contain class names.")

    def predict(self, source: Union[str, List[str]], conf_threshold: float = 0.5, save: bool = False,
                save_dir: str = "./custom_models/runs/predict"):
        """
        Perform predictions on images, videos, or directories of images.
        :param source: Path to an image, video, or folder of images.
        :param conf_threshold: Confidence threshold for predictions.
        :param save: Whether to save the predictions.
        :param save_dir: Directory to save predictions.
        :return: List of predictions containing bounding boxes, labels, and confidence scores.
        """
        if not self.model:
            raise ValueError("No model loaded. Load or train a model first.")

        # Perform predictions
        results = self.model.predict(
            source=source,
            conf=conf_threshold,
            save=save,
            device=self.device,
            save_dir=save_dir
        )

        print(f"Predictions completed. Results saved to: {save_dir}" if save else "Predictions completed.")
        #print("Results structure:", results)

        all_predictions = []

        # Extract predictions for each result
        for result in results:
            image_predictions = []
            boxes = result.boxes.xyxy  # Get bounding box coordinates (tensor)
            confidences = result.boxes.conf  # Get confidence scores (tensor)
            classes = result.boxes.cls  # Get class labels (tensor)
            names = result.names  # Get class name mapping

            for i in range(len(boxes)):
                # Extract data and convert to Python types
                box = boxes[i].tolist()  # Convert tensor to list
                label = int(classes[i].item())  # Get label index as integer
                confidence = float(confidences[i].item())  # Get confidence as float
                image_predictions.append({
                    "box": box,
                    "label": names[label],  # Use the class name from `names`
                    "confidence": confidence
                })
            all_predictions.append(image_predictions)

        return all_predictions

    def evaluate_model(self, data_path: str, save_dir: str = "./custom_models/runs/val"):
        """
        Evaluate the model on a validation dataset.
        :param data_path: Path to the data.yaml file for validation.
        :param save_dir: Directory to save evaluation results.
        :return: Metrics object with evaluation results.
        """
        if not self.model:
            raise ValueError("No model loaded. Load or train a model first.")

        metrics = self.model.val(
            data=data_path,
            save_dir=save_dir,
            device=self.device
        )
        print(f"Evaluation completed. Results saved in: {save_dir}")
        return metrics

    def export_model(self, format: str = "onnx", save_dir: str = "./custom_models/runs/export"):
        """
        Export the model to a specified format.
        :param format: Export format (e.g., 'onnx', 'engine', 'torchscript').
        :param save_dir: Directory to save the exported model.
        """
        if not self.model:
            raise ValueError("No model loaded. Load or train a model first.")

        self.model.export(
            format=format,
            save_dir=save_dir
        )
        print(f"Model exported in {format} format. Saved in: {save_dir}")
