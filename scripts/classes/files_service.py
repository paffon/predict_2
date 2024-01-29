# A class for this project to interface the hard drive
import os
import pickle
from typing import Any

from classes.stack_printer import StackPrinter


def validate_root(folder: str) -> str:
    """
    If this folder exists, return it.
    Else,
    Go up the path from where you are until you hit this folder.
    If found this folder, then return its full path.
    If folder isn't found, then raise an Exception.
    :param folder: The full path or name of a containing folder.
    :return: str
    """

    # Check if the provided path exists
    if os.path.exists(folder):
        return os.path.abspath(folder)

    # Traverse up the path until the folder is found
    # Assuming this function is in the same file
    current_path = os.path.abspath(__file__)

    while current_path != os.path.abspath(os.path.join(current_path, os.pardir)):
        current_path = os.path.abspath(os.path.join(current_path, os.pardir))
        potential_folder = os.path.join(current_path, folder)

        if os.path.exists(potential_folder):
            return os.path.abspath(potential_folder)

    raise Exception(f'Folder "{folder}" not found')


class FilesService:
    def __init__(self, root: str, stack_printer: StackPrinter = StackPrinter()):
        self.root = validate_root(root)
        self.sp = stack_printer

    def save_pickle(self, data: Any, folder: str, file_name: str) -> None:
        """
        A method to save a pickle file to the hard drive.

        :param data: the object to be pickled
        :param folder: the project folder in which to save
        :param file_name: the name of the pickled file, including extension
        :return: None
        """
        if file_name.endswith('.pkl'):
            file_name = file_name[:-4]

        self.sp.print(f'Pickling "{file_name}"...')

        folder_path = os.path.join(self.root, folder)

        file_path = os.path.join(folder_path, file_name + '.pkl')

        # Save the pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def load_pickle(self, folder: str, file_name: str) -> Any:
        """
        A method to load a pickle file from the hard drive.

        :param folder: the project folder from which to load
        :param file_name: the name of the pickled file, including extension
        :return: The un pickled - file itself
        """
        if file_name.endswith('.pkl'):
            file_name = file_name[:-4]

        self.sp.print(f'Getting pickle: "{file_name}"')

        file_path = os.path.join(self.root, folder, file_name + '.pkl')
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_all_files_in_folder(self, folder: str) -> list[str]:
        """
        A method to get all the files in a folder.

        :param folder: the project folder from which to load
        :return: A list of the files in the folder
        """
        self.sp.print(f'Getting all files in folder "{folder}"')

        folder_path = os.path.join(self.root, folder)

        files = os.listdir(folder_path)

        return files
