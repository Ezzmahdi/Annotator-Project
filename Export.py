"""
CVAT Data Export Module

This module provides functionality to export data from CVAT (Computer Vision Annotation Tool)
and convert it into standard formats for both classification and object detection tasks.

Dependencies:
    - os
    - requests
    - zipfile
    - json
    - shutil
    - time
    - pathlib
"""

import os
import requests
import zipfile
import json
import shutil
import time
from shutil import move, rmtree
from pathlib import Path
from typing import Optional, Dict, Any

def export_annotations(
    cvat_url: str,
    headers: Dict[str, str],
    task_id: int,
    output_dir: str,
    annotation_format: str = "COCO 1.1",
    poll_interval: int = 5
) -> Optional[str]:
    """
    Export annotations from a CVAT task.

    Args:
        cvat_url (str): Base URL of the CVAT server
        headers (Dict[str, str]): Request headers containing authentication
        task_id (int): ID of the task to export
        output_dir (str): Directory where exported files will be saved
        annotation_format (str): Format of the exported annotations
        poll_interval (int): Time in seconds between status checks

    Returns:
        Optional[str]: Path to the exported file if successful, None otherwise
    """
    export_url = f"{cvat_url}/api/tasks/{task_id}/dataset"
    params = {"format": annotation_format}

    # Initialize export request
    print(f"Initiating export for Task ID {task_id} with format '{annotation_format}'...")
    response = requests.get(export_url, headers=headers, params=params)
    if response.status_code != 202:
        print(f"Error: Failed to initiate export. {response.status_code} {response.reason}")
        print(f"Response content: {response.text}")
        return None

    rq_id = response.json().get("rq_id")
    if not rq_id:
        print("Error: No request ID returned from the server.")
        return None

    # Poll for export status
    status_url = f"{cvat_url}/api/requests/{rq_id}"
    while True:
        status_response = requests.get(status_url, headers=headers)
        if status_response.status_code != 200:
            print(f"Error: Failed to check status. {status_response.status_code} {status_response.reason}")
            return None

        status_data = status_response.json()
        status = status_data.get("status")
        
        if status == "finished":
            file_url = status_data.get("result_url")
            if not file_url:
                print("Error: Export finished but no file URL provided.")
                return None
            break
        elif status == "failed":
            print(f"Error: Export process failed. Details: {status_data.get('message', 'No details provided.')}")
            return None
        else:
            print(f"Export in progress. Current state: {status}. Retrying in {poll_interval} seconds...")
            time.sleep(poll_interval)

    # Download exported file
    os.makedirs(output_dir, exist_ok=True)
    annotation_file = os.path.join(output_dir, f"task_{task_id}_annotations.zip")
    
    file_response = requests.get(file_url, headers=headers, stream=True)
    if file_response.status_code != 200:
        print(f"Error: Failed to download annotations. {file_response.status_code} {file_response.reason}")
        return None

    with open(annotation_file, "wb") as file:
        for chunk in file_response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Annotations successfully downloaded to {annotation_file}")
    return annotation_file

def unzip_and_convert_annotations_OB(zip_path: str, output_path: str) -> None:
    """
    Convert CVAT object detection annotations to YOLO format.

    Args:
        zip_path (str): Path to the exported ZIP file
        output_path (str): Directory where converted annotations will be saved
    """
    # Extract ZIP contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_dir = os.path.join(output_path, "extracted")
        zip_ref.extractall(extract_dir)

    # Setup paths
    images_dir = os.path.join(extract_dir, "images", "default", "images")
    annotations_path = os.path.join(extract_dir, "annotations", "instances_default.json")
    images_output_dir = os.path.join(output_path, "OutputOB", "images")
    labels_output_dir = os.path.join(output_path, "OutputOB", "labels")
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # Move images to output directory
    for img_file in os.listdir(images_dir):
        move(os.path.join(images_dir, img_file), os.path.join(images_output_dir, img_file))

    # Convert annotations to YOLO format
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Process annotations
    image_annotations = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]
        
        # Convert COCO bbox to YOLO format
        x, y, width, height = bbox
        img_info = next(img for img in data["images"] if img["id"] == image_id)
        img_width = img_info["width"]
        img_height = img_info["height"]
        
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        if img_info["file_name"] not in image_annotations:
            image_annotations[img_info["file_name"]] = []
        image_annotations[img_info["file_name"]].append(
            f"{category_id-1} {x_center:.5f} {y_center:.5f} {norm_width:.5f} {norm_height:.5f}"
        )

    # Save YOLO format annotations
    for img_file, annotations in image_annotations.items():
        label_file = os.path.join(labels_output_dir, f"{Path(img_file).stem}.txt")
        with open(label_file, 'w') as f:
            f.write("\n".join(annotations))
    
    # Cleanup
    rmtree(extract_dir)

def organize_and_swap_images_Class(
    input_json: str,
    output_json: str,
    output_zip: str,
    output_dir: str = "OutputClass"
) -> None:
    """
    Organize and reorganize classification images based on label changes.

    Args:
        input_json (str): Path to input JSON file containing original labels
        output_json (str): Path to output JSON file containing new labels
        output_zip (str): Path to ZIP file containing images
        output_dir (str): Directory where organized images will be saved
    """
    # Load JSON files
    with open(input_json, 'r') as f:
        input_data = json.load(f)
    with open(output_json, 'r') as f:
        output_data = json.load(f)

    # Prepare output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Extract images
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                zip_ref.extract(file, images_dir)

    # Flatten directory structure
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            old_path = os.path.join(root, file)
            new_path = os.path.join(images_dir, file)
            if old_path != new_path:
                shutil.move(old_path, new_path)

    # Create label mappings
    input_label_to_id = {label['name']: idx for idx, label in enumerate(input_data['categories']['label']['labels'])}
    output_id_to_label = {label['name']: idx for idx, label in enumerate(output_data['categories']['label']['labels'])}
    output_label_to_id = {v: k for k, v in output_id_to_label.items()}

    # Create class directories
    for label_name in input_label_to_id.keys():
        os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)

    # Organize images by class
    for item in output_data['items']:
        img_path = os.path.join(images_dir, os.path.basename(item['image']['path']))
        if not os.path.exists(img_path):
            continue

        label_id = item['annotations'][0]['label_id']
        label_name = output_label_to_id[label_id]
        new_label_id = input_label_to_id[label_name]
        new_label_name = [k for k, v in input_label_to_id.items() if v == new_label_id][0]
        
        shutil.move(img_path, os.path.join(output_dir, new_label_name, os.path.basename(img_path)))

    # Cleanup
    if not os.listdir(images_dir):
        os.rmdir(images_dir)

    print(f"Images organized and swapped based on label changes. Output stored in: {output_dir}")


def unzip_file(file_path, output_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def Export(
    cvat_url: str,
    headers: Dict[str, str],
    task_id: int,
    output_dir: str,
    classification: bool = False,
    object_detection: bool = False
) -> None:
    """
    Export and process data from CVAT based on task type.

    Args:
        cvat_url (str): Base URL of the CVAT server
        headers (Dict[str, str]): Request headers containing authentication
        task_id (int): ID of the task to export
        output_dir (str): Directory where exported files will be saved
        classification (bool): Whether this is a classification task
        object_detection (bool): Whether this is an object detection task
    """
    # Export annotations
    # Export annotations
    
    
    # Handle file extraction and processing based on task type
    if classification:

        exported_file = export_annotations(cvat_url, headers, task_id, output_dir, "Datumaro 1.0")
        if exported_file:
            print(f"Annotations exported to: {exported_file}")
        else:
            print("Failed to export annotations.")
            return
        # For classification tasks, unzip and organize the images
        unzip_file(exported_file, 'Output')
        unzip_file(exported_file, 'Output/extra')
        organize_and_swap_images_Class(
            input_json="Output/annotations/default.json",
            output_json="Output/extra/annotations/default.json",
            output_zip=exported_file
        )
    elif object_detection:
        exported_file = export_annotations(cvat_url, headers, task_id, output_dir, "COCO 1.0")
        if exported_file:
            print(f"Annotations exported to: {exported_file}")
        else:
            print("Failed to export annotations.")
            return
        # For object detection tasks, unzip and convert annotations
        unzip_and_convert_annotations_OB(exported_file, output_dir)