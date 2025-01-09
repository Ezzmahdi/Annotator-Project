"""
CVAT API Client Module

This module provides functionality to interact with the CVAT (Computer Vision Annotation Tool) API,
including authentication, project creation, and task management.

Dependencies:
    - requests
"""

import requests
from typing import Optional, List, Dict, Union


def authenticate(cvat_url: str, username: str, password: str) -> Optional[str]:
    """
    Authenticate with CVAT server and retrieve authentication token.

    Args:
        cvat_url (str): Base URL of the CVAT server
        username (str): CVAT username
        password (str): CVAT password

    Returns:
        Optional[str]: Authentication token if successful, None otherwise
    """
    login_url = f"{cvat_url}/api/auth/login"
    auth_data = {"username": username, "password": password}
    response = requests.post(login_url, json=auth_data)
    return response.json().get("key")


def create_project(cvat_url: str, headers: Dict[str, str], project_name: str) -> Optional[int]:
    """
    Create a new project in CVAT.

    Args:
        cvat_url (str): Base URL of the CVAT server
        headers (Dict[str, str]): Request headers containing authentication token
        project_name (str): Name of the project to create

    Returns:
        Optional[int]: Project ID if creation successful, None otherwise
    """
    create_project_url = f"{cvat_url}/api/projects"
    project_data = {"name": project_name}
    response = requests.post(create_project_url, headers=headers, json=project_data)

    project_id = response.json().get("id")
    print(f"Project created with id: {project_id}")
    return project_id


def create_task(
    cvat_url: str,
    token: str,
    task_name: str,
    project_id: Optional[int] = None,
    labels: Optional[List[Dict]] = None
) -> Optional[int]:
    """
    Create a new task in CVAT.

    Args:
        cvat_url (str): Base URL of the CVAT server
        token (str): Authentication token
        task_name (str): Name of the task to create
        project_id (Optional[int]): ID of the project to associate task with
        labels (Optional[List[Dict]]): List of label configurations for the task

    Returns:
        Optional[int]: Task ID if creation successful, None otherwise
    """
    create_task_url = f"{cvat_url}/api/tasks"
    headers = {"Authorization": f"Token {token}"}
    task_data = {"name": task_name}

    if project_id:
        task_data["project_id"] = project_id
    if labels:
        task_data["labels"] = labels

    response = requests.post(create_task_url, headers=headers, json=task_data)

    task_id = response.json().get("id")
    print(f"Task created with id: {task_id}")
    return task_id


def save_credentials_to_file(
    file_path: str,
    token: Optional[str] = None,
    project_id: Optional[int] = None,
    task_id: Optional[int] = None
) -> None:
    """
    Save authentication token and IDs to a file.

    Args:
        file_path (str): Path to the file where credentials will be saved
        token (Optional[str]): Authentication token
        project_id (Optional[int]): Project ID
        task_id (Optional[int]): Task ID

    Raises:
        IOError: If file writing fails
    """
    try:
        with open(file_path, "w") as file:
            if token:
                file.write(f"{token}\n")
            if project_id:
                file.write(f"{project_id}\n")
            if task_id:
                file.write(f"{task_id}\n")
        print(f"Credentials saved to {file_path}")
    except Exception as e:
        print(f"Failed to save credentials to file: {e}")


def initialize_cvat_project(
    cvat_url: str,
    username: str,
    password: str,
    project_name: str,
    task_name: str,
    file_path: str,
    labels: Optional[List[Dict]] = None
) -> None:
    """
    Initialize a CVAT project by authenticating, creating a project and task, and saving credentials.

    Args:
        cvat_url (str): Base URL of the CVAT server
        username (str): CVAT username
        password (str): CVAT password
        project_name (str): Name of the project to create
        task_name (str): Name of the task to create
        file_path (str): Path to save credentials
        labels (Optional[List[Dict]]): List of label configurations for the task
    """
    # Step 1: Authenticate and get the token
    token = authenticate(cvat_url, username, password)
    
    if not token:
        print("Authentication failed.")
        return
    
    # Step 2: Create a project
    headers = {"Authorization": f"Token {token}"}
    project_id = create_project(cvat_url, headers, project_name)
    
    if not project_id:
        print("Project creation failed.")
        return
    
    # Step 3: Create a task
    task_id = create_task(cvat_url, token, task_name, labels=labels)
    
    if not task_id:
        print("Task creation failed.")
        return
    
    # Step 4: Save credentials to file
    save_credentials_to_file(file_path, token=token, project_id=project_id, task_id=task_id)