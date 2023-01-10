
import os
import shutil
import json
from typing import Any

class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_json(self, data: Any, filename: str) -> None:
        """Save the given data as a JSON file with the given filename in the data directory."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_json(self, filename: str) -> Any:
        """Load and return the data from the JSON file with the given filename in the data directory."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_data(self, data: Any, filename: str) -> None:
        """Save the given data as a plain text file with the given filename in the data directory."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            f.write(data)
    
    def load_data(self, filename: str) -> Any:
        """Load and return the data from the plain text file with the given filename in the data directory."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return f.read()
    
    def create_folder(self, folder_name: str) -> None:
        """Create a new folder with the given name in the data directory."""
        folder_path = os.path.join(self.data_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    
    def delete_folder(self, folder_name: str) -> None:
        """Delete the folder with the given name in the data directory."""
        folder_path = os.path.join(self.data_dir, folder_name)
        shutil.rmtree(folder_path)
    
    def folder_exists(self, folder_name: str) -> bool:
        """Return True if a folder with the given name exists in the data directory, False otherwise."""
        folder_path = os.path.join(self.data_dir, folder_name)
        return os.path.isdir(folder_path)

