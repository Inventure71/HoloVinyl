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