import glob
import os

import psutil
import torch


def get_optimal_workers(device: str):
    """
    Determines the best number of workers for data loading based on CPU cores and the training device.

    :param device: The training device ('cuda', 'mps', 'cpu')
    :return: Optimal number of workers
    """
    num_cores = psutil.cpu_count(logical=True)  # Total logical cores
    num_physical_cores = psutil.cpu_count(logical=False)  # Physical cores

    # Define base worker count based on CPU cores
    base_workers = min(8, num_physical_cores)  # Use physical cores (max 8)

    # Adjust based on device type
    if device == "cuda":
        optimal_workers = max(2, base_workers // 2)  # Reduce to prevent VRAM saturation
    elif device == "mps":
        optimal_workers = min(base_workers, 4)  # macOS handles threading well
    else:  # CPU
        optimal_workers = min(8, num_cores // 2)  # Use half the logical cores (avoid CPU overload)

    print(f"Optimal workers for {device}: {optimal_workers}")
    return optimal_workers


def get_best_cache_option():
    """
    Determines the best caching method for YOLO training based on available RAM and dataset size.

    :param dataset_path: Path to the dataset (expects images in `images/train/` and `images/val/`).
    :return: Best cache option ('ram', 'disk', or False).
    """
    # Get available system memory
    total_ram = psutil.virtual_memory().available  # Free RAM in bytes

    # Estimate dataset size (sum of all image file sizes)
    def estimate_dataset_size(image_folder):
        image_files = glob.glob(os.path.join(image_folder, "*"))
        return sum(os.path.getsize(f) for f in image_files if os.path.isfile(f))

    # Estimate total dataset size (train + val)
    train_size = estimate_dataset_size("custom_models/yolo_dataset/train/images")
    val_size = estimate_dataset_size("custom_models/yolo_dataset/val/images")
    dataset_size = train_size + val_size

    # Add buffer (factor of 1.5) for additional memory usage
    estimated_needed_memory = dataset_size * 1.5

    print(f"Dataset Size: {dataset_size / 1e9:.2f} GB, Available RAM: {total_ram / 1e9:.2f} GB")

    # Choose cache option based on memory constraints
    if estimated_needed_memory < total_ram * 0.8:  # If dataset fits in 80% of available RAM
        print("Using cache='ram' for faster training.")
        return "ram"
    elif dataset_size < 100 * 1e9:  # If dataset is large but manageable on SSD (<100GB)
        print("Using cache='disk' to avoid RAM overload.")
        return "disk"
    else:
        print("Cache disabled to prevent system overload.")
        return False  # Avoid caching if the dataset is too large


def get_optimal_batch_size():
    """
    Dynamically determines the best batch size based on available system resources.
    Adjusts settings for CPU, MPS (macOS), and CUDA.
    """

    data = psutil.virtual_memory()
    print(data)
    # get total ram
    usable_ram = data.total * 0.65
    print("Can use up to:", usable_ram)

    if torch.backends.mps.is_available():
        print("Using MPS for YOLO inference.")
        device = 'mps'
    elif torch.cuda.is_available():
        print("Using CUDA for YOLO inference.")
        device = 'cuda'
        vram = torch.cuda.get_device_properties(0).total_memory  # Get total VRAM
        available_vram = torch.cuda.memory_reserved(0)  # Available VRAM
        usable_ram = min(usable_ram, available_vram)  # Prioritize VRAM usage
        print(f"Using CUDA, VRAM: {vram / 1e9:.2f} GB, Available: {available_vram / 1e9:.2f} GB")
    else:
        print("Using CPU for YOLO inference.")
        device = 'cpu'


    # Define batch size heuristics based on memory
    if usable_ram > 16 * 1e9:  # 16GB+ memory
        batch_size = 64
    elif usable_ram > 8 * 1e9:  # 8-16GB memory
        batch_size = 32
    elif usable_ram > 4 * 1e9:  # 4-8GB memory
        batch_size = 16
    else:
        batch_size = 8  # Low-memory fallback

    # Adjust subdivisions to balance batch size
    subdivisions = 1 if batch_size >= 32 else 2 if batch_size >= 16 else 4

    return batch_size, subdivisions, device


def find_parameters():
    """
    Finds the optimal training parameters based on the system configuration.
    Includes batch size, subdivisions, image size, workers, and augmentation tweaks.
    """

    batch_size, subdivisions, device = get_optimal_batch_size()

    # Adjust image size dynamically
    #img_size = 640 if batch_size >= 32 else 416 if batch_size >= 16 else 320  # Lower for weaker hardware

    cache_type = get_best_cache_option()

    # Optimize number of workers for multi-threaded data loading
    num_workers = get_optimal_workers(device)

    dic = {
        "batch_size": batch_size,
        "subdivisions": subdivisions,
        "img_size": 640,
        "num_workers": num_workers,
        "use_cache": cache_type,
        "device": device,
    }
    print(dic)

    # Return optimized training settings
    return dic