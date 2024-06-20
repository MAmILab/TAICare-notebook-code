import os
import logging

class DirectoryManager:
    @staticmethod
    def change_directory(path):
        """
        Changes the current working directory to the specified path and logs the operation.
        
        This method is particularly useful for scripts that need to operate in a specific directory,
        especially when dealing with file I/O operations that assume files are located in the current directory.
        
        Args:
            path (str): The path to set as the current working directory.
        """
        try:
            os.chdir(path)
            logging.info(f"Changed directory to {path}")
        except Exception as e:
            logging.error(f"Error changing directory: {e}")
            raise
