import time

from utils.database_handler_V3 import create_or_update_yolo_dataset
from utils.yolo_handler import YOLOHandler


if __name__ == "__main__":
    # Initialize the YOLO handler (load a pretrained model or start fresh)
    yolo_handler = YOLOHandler(model_path="yolo11n.pt")  # Pretrained YOLOv8 model

    start_time = time.time()
    class_dirs = {
        "postit": "raw_images/postit",
        "pen": "raw_images/pen"
    }

    create_or_update_yolo_dataset(
        class_directories=class_dirs,
        output_directory="yolo_dataset",
        target_samples_per_class=70
    )
    print(f"Dataset creation completed in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()

    # Train on a custom dataset
    yolo_handler.train_model(
        data_path="yolo_dataset/dataset.yaml",
        model_type="yolo11n.pt",  # Small model
        epochs=50,
        batch_size=16,
        img_size=640
    )
    print(f"Model training completed in {time.time() - start_time:.2f} seconds.")
    
    # Export to ONNX format for deployment
    #yolo_handler.export_model(format="onnx")
    #print("Model training completed.")

    """
    # Load a retrained model
    yolo_handler.load_model(model_path="runs/detect/train/weights/best.pt")

    # Perform predictions on an image or folder of images
    results = yolo_handler.predict(
        source="raw_images/img.png", #path/to/image_or_folder
        conf_threshold=0.5,
        save=True,
        save_dir="./runs/predict"
    )

    # Evaluate on validation dataset
    #metrics = yolo_handler.evaluate_model(data_path="yolo_dataset/dataset.yaml")"""



