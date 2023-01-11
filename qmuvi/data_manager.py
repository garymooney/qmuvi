
import os
import shutil
import json
import glob
from typing import Any, AnyStr


class DataManager:
    def __init__(self, data_dir: str, default_name: str = 'data'):
        """ Create a new DataManager object. The data directory is created if it does not exist.
        Args:
            data_dir: The directory to save and load data from.
            default_name: The default name to use when saving and loading data.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.default_name = os.path.splitext(default_name)[0]

    def save_json(self, data: Any, filename: str=None) -> None:
        """ Save the given data as a JSON file with the given filename in the data directory.
        Args:
            data: The data to save.
            filename: The name of the file to save the data to. If None, the default name is used.
        """
        if filename is None:
            filename = self.default_name + ".json"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_json(self, filename: str=None) -> Any:
        """ Load and return the data from the JSON file with the given filename in the data directory.
        Args:
            filename: The name of the file to load the data from.
        """
        if filename is None:
            filename = self.default_name + ".json"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)

    def save_data(self, data: Any, filename: str=None) -> None:
        """ Save the given data as a plain text file with the given filename in the data directory.
        Args:
            data: The data to save.
            filename: The name of the file to save the data to. If None, the default name is used.
        """
        if filename is None:
            filename = self.default_name + ".txt"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            f.write(data)

    def load_data(self, filename: str=None) -> Any:
        """ Load and return the data from the plain text file with the given filename in the data directory.
        Args:
            filename: The name of the file to load the data from.
        """
        if filename is None:
            filename = self.default_name + ".txt"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return f.read()

    def save_midi(self, mid, filename: str=None):
        if filename is not None:
            filename = os.path.splitext(filename)[0] + ".mid"
        else:
            filename = self.default_name + ".mid"
        mid.save(os.path.join(self.data_dir, filename))

    def create_folder(self, folder_name: str) -> None:
        """Create a new folder with the given name in the data directory (appends to data_dir)."""
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

    def glob(self, subpathname: AnyStr, *, recursive: bool=...):
        """Return a list of subpaths of data_dir matching a pathname pattern.
        Uses glob.glob method The pattern may contain simple shell-style wildcards.
        """
        return glob.glob(os.path.join(self.data_dir, subpathname), recursive=recursive)

    def get_default_file_pathname(self):
        """Return the default file pathname (with no extension) for the data directory."""
        return os.path.join(self.data_dir, self.default_name)

def extract_natural_number_from_string_end(s: str, zero_if_none = False) -> int:
    if s is None or len(s) == 0:
        return None

    string_iter = len(s)-1
    end_number_digits = []
    for i in range(len(s)):
        if not s[string_iter].isdigit():
            break
        end_number_digits.append(s[string_iter])
        string_iter -= 1
    if len(end_number_digits) > 0:
        end_number = int(''.join(end_number_digits[::-1]))
    else:
        end_number = 0 if zero_if_none else None
    return end_number

