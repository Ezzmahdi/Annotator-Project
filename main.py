import os
from typing import Tuple

from Auth import initialize_cvat_project
from Import import Import, extract_classification_labels, extract_object_detection_labels
from Export import Export

# Configuration constants
CVAT_URL = "http://localhost:8080"
USERNAME = "MahdiEzz"
PASSWORD = "1234Mahdoodi"

# Directory paths
CLASSIFICATION_IMAGES_DIR = "fruits/train"
OBJECT_DETECTION_IMAGES_DIR = "fruitOB/train/images"
ANNOTATIONS_DIR = "fruitOB/train/labels"
OUTPUT_DIR = "Output"
ANNOTATION_FORMAT = "COCO 1.0"
CREDENTIALS_FILE = "credentials.txt"

# Object detection class names
OB_NAMES = [
    "apple", "apricot", "banana", "grapes", "kiwi", "melon", "cherry", "potato", "cat"
]

def get_project_settings(project_type: str) -> Tuple[bool, bool, str, str, str]:
    """
    Determine project settings based on project type.
    Returns: (classification, object_detection, images_dir, task_name, project_name)
    """
    if project_type == "1":
        return True, False, CLASSIFICATION_IMAGES_DIR, "Classification Task", "Classification Project"
    else:
        return False, True, OBJECT_DETECTION_IMAGES_DIR, "ObjectDetection Task", "ObjectDetection Project"

def load_credentials() -> Tuple[str, str, str]:
    """Load and return credentials from file."""
    with open(CREDENTIALS_FILE, "r") as file:
        return (
            file.readline().strip(),  # token
            file.readline().strip(),  # project_id
            file.readline().strip()   # task_id
        )

def main():
    # Initialize project variables
    classification = object_detection = False
    labels = None
    
    try:
        # Get project type from user
        project_type = input("Is this a 1. classification or 2. object detection project: ").strip()
        
        # Validate input
        if project_type not in ["1", "2"]:
            raise ValueError("Invalid project type. Please enter '1' or '2'.")
        
        # Set project configuration based on type
        classification, object_detection, images_dir, task_name, project_name = (get_project_settings(project_type))
        
        # Extract appropriate labels based on project type
        labels = (
            extract_classification_labels(CLASSIFICATION_IMAGES_DIR)
            if classification
            else extract_object_detection_labels(ANNOTATIONS_DIR)
        )
        
        # Initialize CVAT project and save credentials
        initialize_cvat_project(
            CVAT_URL, USERNAME, PASSWORD, task_name, project_name, CREDENTIALS_FILE, labels=labels
        )
        
        # Load credentials for API access
        token, project_id, task_id = load_credentials()
        headers = {"Authorization": f"Token {token}"}
        
        # Import data to CVAT
        success = Import(
            CVAT_URL, USERNAME, PASSWORD, headers,
            project_id, int(task_id), OUTPUT_DIR,
            classification=classification,
            object_detection=object_detection,
            images_dir=images_dir,
            annotations_dir=ANNOTATIONS_DIR
        )
        
        if success:
            print("Project data imported successfully.")
        
        # Wait for user to complete annotations
        input("\nPress ENTER after editing your annotations in CVAT to export them...")
        
        # Export annotations from CVAT
        Export(
            CVAT_URL, headers, int(task_id), OUTPUT_DIR,
            classification=classification,
            object_detection=object_detection
        )
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()