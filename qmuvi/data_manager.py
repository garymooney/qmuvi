"""This module is used to manage data generated within the qmuvi package."""

import glob
import json
import os
import shutil
from typing import Any, AnyStr


class DataManager:
    """Used to manage saving and loading within a target data directory."""

    def __init__(self, data_dir: str, default_name: str = "data"):
        """Create a new DataManager object. The data directory is created if it does not exist.

        Parameters
        ----------
            data_dir
                The directory to save and load data from.
            default_name
                The default name to use when saving and loading data.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.default_name = os.path.splitext(default_name)[0]

    def save_json(self, data: Any, filename: str = None) -> None:
        """Save the given data as a JSON file with the given filename in the data directory.

        Parameters
        ----------
            data
                The data to save.
            filename
                The name of the file to save the data to. If None, the default name is used.
        """
        if filename is None:
            filename = self.default_name + ".json"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_json(self, filename: str = None) -> Any:
        """Load and return the data from the JSON file with the given filename in the data directory.

        Parameters
        ----------
            filename
                The name of the file to load the data from.

        Returns
        -------
            The data loaded from the JSON file.
        """
        if filename is None:
            filename = self.default_name + ".json"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as f:
            return json.load(f)

    def save_data(self, data: Any, filename: str = None) -> None:
        """Save the given data as a plain text file with the given filename in the data directory.

        Parameters
        ----------
            data
                The data to save.
            filename
                The name of the file to save the data to. If None, the default name is used.
        """
        if filename is None:
            filename = self.default_name + ".txt"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w") as f:
            f.write(data)

    def load_data(self, filename: str = None) -> Any:
        """Load and return the data from the plain text file with the given filename in the data directory.

        Parameters
        ----------
            filename
                The name of the file to load the data from.
        """
        if filename is None:
            filename = self.default_name + ".txt"
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as f:
            return f.read()

    def create_folder(self, folder_name: str) -> None:
        """Create a new folder with the given name in the data directory (appends to data_dir).

        Parameters
        ----------
            folder_name
                The name of the folder to create.
        """
        folder_path = os.path.join(self.data_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

    def delete_folder(self, folder_name: str) -> None:
        """Delete the folder with the given name in the data directory.

        Parameters
        ----------
            folder_name
                The name of the folder to delete.
        """
        folder_path = os.path.join(self.data_dir, folder_name)
        shutil.rmtree(folder_path)

    def folder_exists(self, folder_name: str) -> bool:
        """Return True if a folder with the given name exists in the data directory, False otherwise.

        Parameters
        ----------
            folder_name
                The name of the folder to check for.

        Returns
        -------
            True if a folder with the given name exists in the data directory, False otherwise.
        """
        folder_path = os.path.join(self.data_dir, folder_name)
        return os.path.isdir(folder_path)

    def glob(self, subpathname: AnyStr, *, recursive: bool = ...) -> list[str]:
        """Return a list of subpaths of data_dir matching a pathname pattern.

        Uses the glob.glob method. The pattern may contain simple shell-style wildcards.

        Parameters
        ----------
            subpathname
                The pathname pattern to match.
            recursive
                If True, the pattern “**” will match any files and zero or more directories and subdirectories.

        Returns
        -------
            A list of subpaths of data_dir matching a pathname pattern.
        """
        return glob.glob(os.path.join(self.data_dir, subpathname), recursive=recursive)

    def get_default_file_pathname(self) -> str:
        """Return the default file pathname (with no extension) for the data directory.

        Returns
        -------
            The default file pathname (with no extension) for the data directory.
        """
        return os.path.join(self.data_dir, self.default_name)

    def get_path(self, filename: str) -> str:
        """Return the full path to the given filename in the data directory.

        Parameters
        ----------
            filename
                The name of the file to get the path to.

        Returns
        -------
            The full path to the given filename in the data directory.
        """
        return os.path.join(self.data_dir, filename)

    def remove_files(self, filename: str) -> None:
        """Remove all files with the given name in the data directory.

        Parameters
        ----------
            filename
                The name of the files to remove.
        """
        for file in glob.glob(os.path.join(self.data_dir, filename)):
            os.remove(file)


def extract_natural_number_from_string_end(s: str, zero_if_none=False) -> int:
    """Extract the last number of 1 or more digits from the end of the given string.

    Parameters
    ----------
    s
        The string to extract the number from.
    zero_if_none, optional
        If True, return 0 if no number is found at the end of the string, otherwise return None. Default: False.

    Returns
    -------
        The last number of 1 or more digits from the end of the given string, or 0 if no number is found and zero_if_none is True.
    """
    if s is None or len(s) == 0:
        return None

    string_iter = len(s) - 1
    end_number_digits = []
    for i in range(len(s)):
        if not s[string_iter].isdigit():
            break
        end_number_digits.append(s[string_iter])
        string_iter -= 1
    if len(end_number_digits) > 0:
        end_number = int("".join(end_number_digits[::-1]))
    else:
        end_number = 0 if zero_if_none else None
    return end_number


def get_unique_pathname(base_name: str, location: str) -> str:
    """Get a unique pathname in the working directory based on the base_name.

    If a file with the name already exists in the working directory, a number is appended to the end of the name.
    E.g. If the folder "test" already exists, the number "1" is appended to get "test-1". If "test-1" already exists,
    the number is incremented to get "test-2" and so on.

    Parameters
    ----------
        base_name
            The base name for the new pathname.
        location
            The directory location to create the new pathname in.

    Returns
    -------
        The name of the new pathname.
    """
    # Create output folder with a different name to existing folders
    folder_names = [dir for dir in glob.glob(os.path.join(location, base_name + "*")) if os.path.isdir(dir)]

    if len(folder_names) == 0:
        output_folder_name = base_name
    else:
        folder_ending_numbers = [extract_natural_number_from_string_end(folder, zero_if_none=True) for folder in folder_names]
        output_folder_name = base_name + "-" + str(max(folder_ending_numbers) + 1)

    output_path = os.path.join(location, output_folder_name)
    return output_path
