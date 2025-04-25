"""
This script renames JSON files in a specified directory by removing the suffix added by the Naver Clova Speech step.
The script takes the directory path as an argument and renames all JSON files in that directory.
The script uses the os and json libraries to handle file operations and JSON data.
"""

import os
import json
import sys

from tqdm import tqdm


def rename_json_files(directory):
    """
    This function renames JSON files in the specified directory by removing the suffix added by the Naver Clova Speech step.

    :param directory: The path to the directory containing the JSON files.
    """
    # Iterate through all files in the directory
    for filename in tqdm(os.listdir(directory), desc="Renaming JSON files", unit="file", total=len(os.listdir(directory))):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)

            # Load the JSON data from the file
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            # Print the filename
            # print(f"filename: {filename}")

            # split the filename by underscore
            parts = filename.split('_')
            # print the parts
            # print(f"parts: {parts}")
            # remove the last two parts from the filename
            part = parts[:-2]
            # print(f"part: {part}")
            # join the parts back together
            new_filename = '_'.join(part) + '.json'
            # print the new filename
            # print(f"new_filename: {new_filename}")
            # Construct the new file path
            new_file_path = os.path.join(directory, new_filename)
            # print(f"new_file_path: {new_file_path}")
            # print("*" * 50)

            # Save the JSON data to the new file
            with open(new_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)

            # Remove the old file
            os.remove(file_path)
            # print(f'Renamed {filename} to {new_filename}')


if __name__ == "__main__":
    #set the path to the directory containing the JSON files
    directory_path = '../data/transcripts/non_vishing'

    # Call the function to rename JSON files
    rename_json_files(directory_path)
