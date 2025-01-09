"""
CVAT Data Import Module

This module provides functionality to import both classification and object detection 
data into CVAT (Computer Vision Annotation Tool). It handles image uploads, annotation 
creation, and data formatting for both task types.

Dependencies:
    - os
    - requests
    - zipfile
    - json
    - PIL
    - cvat_sdk
"""

import os
import requests
import zipfile
import json
from PIL import Image
from cvat_sdk import make_client
from typing import List, Dict, Union, Optional, Any

# Predefined object detection class names
OB_NAMES = ["apple", "apricot", "banana", "grapes", "kiwi", "melon", "cherry", "potato", "cat"]

def upload_images_to_task(cvat_url: str, headers: Dict[str, str], task_id: int, coco_zip_path: str) -> None:
    """
    Upload zipped images to a CVAT task.

    Args:
        cvat_url (str): Base URL of the CVAT server
        headers (Dict[str, str]): Request headers containing authentication
        task_id (int): ID of the target task
        coco_zip_path (str): Path to the zip file containing images
    """
    url = f"{cvat_url}/api/tasks/{task_id}/data"

    with open(coco_zip_path, "rb") as zip_file:
        files = {"client_files[0]": zip_file}
        data = {"image_quality": 70}
        response = requests.post(url, headers=headers, data=data, files=files)
        print(f"COCO data uploaded successfully to task {task_id}.")

# Classification-related functions
def extract_classification_labels(images_dir: str) -> List[Dict[str, List]]:
    """
    Extract classification labels from directory structure.

    Args:
        images_dir (str): Directory containing class-specific subdirectories

    Returns:
        List[Dict[str, List]]: List of label configurations
    """
    labels = []
    for folder_name in os.listdir(images_dir):
        if os.path.isdir(os.path.join(images_dir, folder_name)):
            labels.append({"name": folder_name, "attributes": []})
    return labels

def create_classification_zip(images_dir: str, output_zip_path: str) -> None:
    """
    Create a zip file containing classification images with their directory structure.

    Args:
        images_dir (str): Source directory containing class-specific subdirectories
        output_zip_path (str): Path where the zip file will be created
    """
    with zipfile.ZipFile(output_zip_path, "w") as zipf:
        for folder_name in os.listdir(images_dir):
            folder_path = os.path.join(images_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(folder_path, file_name)
                        relative_path = os.path.relpath(file_path, images_dir)
                        zipf.write(file_path, arcname=f"images/{relative_path}")
    print("Classification ZIP created successfully.")

def create_classification_annotations(base_dir: str, output_dir: str) -> None:
    """
    Create CVAT-compatible classification annotations.

    Args:
        base_dir (str): Directory containing classified images
        output_dir (str): Directory where annotation files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_file = os.path.join(output_dir, "annotations.zip")
    temp_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(temp_annotations_dir, exist_ok=True)
    
    annotations = {
        "info": {},
        "categories": {"label": {"labels": [], "attributes": ["occluded"]}},
        "items": []
    }

    # Process each class directory
    label_mapping = {}
    for label_id, label_name in enumerate(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        label_mapping[label_name] = label_id
        annotations["categories"]["label"]["labels"].append({
            "name": label_name,
            "parent": "",
            "attributes": []
        })

        # Process images in each class
        for img_name in os.listdir(label_path):
            if not img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            img_path = os.path.join(label_path, img_name)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue

            # Add image annotation
            annotations["items"].append({
                "id": os.path.relpath(img_path, base_dir).replace("\\", "/"),
                "annotations": [{
                    "id": 0,
                    "type": "label",
                    "attributes": {},
                    "group": 0,
                    "label_id": label_mapping[label_name]
                }],
                "attr": {"frame": len(annotations["items"])},
                "image": {
                    "path": os.path.relpath(img_path, base_dir).replace("\\", "/"),
                    "size": [height, width]
                }
            })

    # Save and zip annotations
    json_file = os.path.join(temp_annotations_dir, "default.json")
    with open(json_file, 'w') as f:
        json.dump(annotations, f, indent=4)

    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(json_file, os.path.relpath(json_file, output_dir))

    # Cleanup temporary files
    os.remove(json_file)
    os.rmdir(temp_annotations_dir)
    print(f"Annotations zipped as {zip_file}")

# Object detection-related functions
def extract_object_detection_labels(labels_dir: str) -> List[Dict[str, List]]:
    """
    Extract object detection labels from annotation files.

    Args:
        labels_dir (str): Directory containing YOLO format label files

    Returns:
        List[Dict[str, List]]: List of label configurations
    """
    labels_set = set()
    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(labels_dir, label_file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    labels_set.add(class_id)

    return [
        {"name": OB_NAMES[class_id], "attributes": []}
        for class_id in sorted(labels_set) if class_id < len(OB_NAMES)
    ]

def create_object_detection_zip(images_dir: str, output_zip_path: str) -> None:
    """
    Create a zip file containing object detection images.

    Args:
        images_dir (str): Directory containing images
        output_zip_path (str): Path where the zip file will be created
    """
    with zipfile.ZipFile(output_zip_path, "w") as zipf:
        for file_name in os.listdir(images_dir):
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(images_dir, file_name)
                zipf.write(image_path, arcname=f"images/{file_name}")
    print("Object detection ZIP created successfully.")

def create_image_metadata(image_dir: str) -> List[Dict[str, Union[int, str]]]:
    """
    Create metadata for images in a directory.

    Args:
        image_dir (str): Directory containing images

    Returns:
        List[Dict[str, Union[int, str]]]: List of image metadata
    """
    metadata = []
    for idx, filename in enumerate(os.listdir(image_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            with Image.open(os.path.join(image_dir, filename)) as img:
                metadata.append({
                    "id": idx + 1,
                    "width": img.size[0],
                    "height": img.size[1],
                    "file_name": filename
                })
    return metadata

def create_object_detection_annotations(
    labels_dir: str,
    image_metadata: List[Dict[str, Any]],
    output_zip_path: str = "coco_annotations.zip"
) -> Dict[str, Any]:
    """
    Create COCO format annotations for object detection.

    Args:
        labels_dir (str): Directory containing YOLO format labels
        image_metadata (List[Dict[str, Any]]): List of image metadata
        output_zip_path (str): Path where annotations will be saved

    Returns:
        Dict[str, Any]: COCO format annotations
    """
    coco_format = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "",
                "url": "", "version": "", "year": ""},
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Process annotations
    categories_set = set()
    annotations = []
    annotation_id = 1

    for metadata in image_metadata:
        label_file = os.path.join(labels_dir, f"{os.path.splitext(metadata['file_name'])[0]}.txt")
        if not os.path.exists(label_file):
            print(f"Warning: Annotation file not found for {metadata['file_name']}")
            continue

        coco_format["images"].append({
            "id": metadata['id'],
            "width": metadata['width'],
            "height": metadata['height'],
            "file_name": metadata['file_name'],
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

        # Process YOLO format annotations
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Skipping malformed line in {label_file}: {line}")
                    continue

                class_id = int(parts[0])
                if class_id >= len(OB_NAMES):
                    print(f"Skipping invalid class ID {class_id} in {label_file}")
                    continue

                categories_set.add(class_id)
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert YOLO to COCO format
                img_width = metadata['width']
                img_height = metadata['height']
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                bbox_width = width * img_width
                bbox_height = height * img_height

                annotations.append({
                    "id": annotation_id,
                    "image_id": metadata['id'],
                    "category_id": class_id + 1,
                    "segmentation": [],
                    "area": bbox_width * bbox_height,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "iscrowd": 0,
                    "attributes": {"occluded": False, "rotation": 0.0}
                })
                annotation_id += 1

    # Create categories
    coco_format["categories"] = [
        {"id": class_id + 1, "name": OB_NAMES[class_id], "supercategory": ""}
        for class_id in sorted(categories_set) if class_id < len(OB_NAMES)
    ]
    coco_format["annotations"] = annotations

    # Save annotations
    temp_dir = "temp_annotations"
    os.makedirs(temp_dir, exist_ok=True)
    json_file_path = os.path.join(temp_dir, "instances_default.json")

    with open(json_file_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        zipf.write(json_file_path, arcname="annotations/instances_default.json")

    # Cleanup
    os.remove(json_file_path)
    os.rmdir(temp_dir)

    print(f"COCO annotations saved to {output_zip_path}")
    return coco_format

def upload_annotation(
    cvat_server: str,
    port: int,
    username: str,
    password: str,
    task_id: int,
    annotations_filename: str,
    annotation_format: str
) -> bool:
    """
    Upload annotations to a CVAT task.

    Args:
        cvat_server (str): CVAT server address
        port (int): Server port
        username (str): CVAT username
        password (str): CVAT password
        task_id (int): Target task ID
        annotations_filename (str): Path to annotations file
        annotation_format (str): Format of annotations ('COCO 1.0' or 'Datumaro 1.0')

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        with make_client(cvat_server, port=port, credentials=(username, password)) as client:
            print(f"Connecting to CVAT server at {cvat_server}:{port}...")
            task = client.tasks.retrieve(task_id)
            task.import_annotations(annotation_format, annotations_filename, status_check_period=0.1)
            print(f"Annotations successfully uploaded for Task ID: {task_id}")
            return True
    except Exception as e:
        print(f"Error uploading annotations: {e}")
        return False

def Import(
    cvat_url: str,
    username: str,
    password: str,
    headers: Dict[str, str],
    project_id: int,
    task_id: int,
    output_dir: str,
    classification: bool = False,
    object_detection: bool = False,
    images_dir: Optional[str] = None,
    annotations_dir: Optional[str] = None
) -> bool:
    """
    Import data into CVAT for either classification or object detection tasks.

    Args:
        cvat_url (str): Base URL of the CVAT server
        username (str): CVAT username
        password (str): CVAT password
        headers (Dict[str, str]): Request headers
        project_id (int): Project ID
        task_id (int): Task ID
        output_dir (str): Directory for temporary files
        classification (bool): Whether this is a classification task
        object_detection (bool): Whether this is an object detection task
        images_dir (Optional[str]): Directory containing images
        annotations_dir (Optional[str]): Directory containing annotations

    Returns:
        bool: True if import successful, False otherwise
    """
    if classification:
        # Create classification data zip and upload
        coco_zip_path = os.path.join(output_dir, "classification_coco_data.zip")
        create_classification_zip(images_dir, coco_zip_path)
        upload_images_to_task(cvat_url, headers, task_id, coco_zip_path)

        create_classification_annotations(images_dir, "Output")
        success = upload_annotation("localhost", 8080, username, password, task_id, "Output/annotations.zip", "Datumaro 1.0")
        return success

    elif object_detection:
        # Create object detection data zip and upload
        coco_zip_path = os.path.join(output_dir, "object_detection_coco_data.zip")
        create_object_detection_zip(images_dir, coco_zip_path)
        upload_images_to_task(cvat_url, headers, task_id, coco_zip_path)

        # Create metadata and annotations for object detection
        metadata = create_image_metadata(images_dir)
        create_object_detection_annotations(annotations_dir, metadata)

        success = upload_annotation("localhost", 8080, username, password, task_id, "coco_annotations.zip", "COCO 1.0")
        return success

    else:
        print("Please specify either 'classification' or 'object_detection'.")
        return False



