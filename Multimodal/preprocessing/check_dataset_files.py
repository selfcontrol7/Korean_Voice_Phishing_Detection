import os


def get_all_files(folder):
    """
    Recursively collects all file paths (with extensions) relative to the given folder.

    Args:
        folder (str): The path to the folder.

    Returns:
        list: A list of file paths relative to the folder.
    """
    file_list = []
    for root, dirs, files in os.walk(folder):
        rel_dir = os.path.relpath(root, folder)
        for file in files:
            # Build the relative file path
            rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
            file_list.append(rel_file)
    return file_list


def remove_extensions(file_list, extensions):
    """
    Removes specified extensions from file names.

    Args:
        file_list (list): List of file paths.
        extensions (list): List of extensions to remove.

    Returns:
        list: List of file paths with extensions removed.
    """
    modified_list = []
    for file in file_list:
        # Check if the file has any of the specified extensions
        if any(file.endswith(ext) for ext in extensions):
            # Remove the extension from the file name
            base_name = os.path.splitext(file)[0]
            # Check for double extensions (e.g., .wav.txt)
            if any(base_name.endswith(ext) for ext in extensions):
                base_name = os.path.splitext(base_name)[0]
            modified_list.append(base_name)
        else:
            # If it doesn't have any of the specified extensions, keep it as is
            modified_list.append(file)
    return modified_list


def find_missing_files(list_a, list_b, folder_a_name, folder_b_name):
    """
    Finds and reports files that are in list_b but not in list_a.

    Args:
        list_a (list): First list of files.
        list_b (list): Second list of files.
        folder_a_name (str): Name of the first folder for reporting.
        folder_b_name (str): Name of the second folder for reporting.
    """
    missing_in_a = [f for f in list_b if f not in list_a]

    print(f"Total number of files in {folder_a_name}:", len(list_a))
    print(f"Total number of files in {folder_b_name}:", len(list_b))
    print(f"Total number of files missing in {folder_a_name}:", len(missing_in_a))

    if missing_in_a:
        print(f"Files in {folder_b_name} that are missing in {folder_a_name}:")
        for file in sorted(missing_in_a):
            print(file)
    else:
        print(f"{folder_a_name} has all the files present in {folder_b_name}.")


def compare_folders(folder_a, folder_b, print_all_files=False):
    """
    Compares two folders to find missing files in either folder.

    Args:
        folder_a (str): Path to the first folder.
        folder_b (str): Path to the second folder.
        print_all_files (bool): Whether to print all files in both folders.
    """
    print(f"Comparing folders:\n  A: {folder_a}\n  B: {folder_b}")

    # Get all files from both folders
    files_a = get_all_files(folder_a)
    files_b = get_all_files(folder_b)

    print(f"Raw file count - Folder A: {len(files_a)}, Folder B: {len(files_b)}")
    print("=" * 50)

    # Define extensions to remove
    file_ext = ['.wav.txt', '.mp3.txt', '.wav', '.mp3', '.txt', '.mp4', '.json']

    # Remove extensions
    files_a_modified = remove_extensions(files_a, file_ext)
    files_b_modified = remove_extensions(files_b, file_ext)

    # Print all files if requested
    if print_all_files:
        print("Modified files in Folder A (with extensions removed):")
        for file in sorted(files_a_modified):
            print(file)

        print("=" * 50)

        print("Modified files in Folder B (with extensions removed):")
        for file in sorted(files_b_modified):
            print(file)

        print("=" * 50)

    # Find files missing in Folder A
    print("Checking for files missing in Folder A:")
    find_missing_files(files_a_modified, files_b_modified, "Folder A", "Folder B")

    print("=" * 50)

    # Find files missing in Folder B
    print("Checking for files missing in Folder B:")
    find_missing_files(files_b_modified, files_a_modified, "Folder B", "Folder A")


if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    folder_a = '../data/audio/non_vishing'
    folder_b = '../data/transcripts/non_vishing'

    # Set to True if you want to print all files in both folders
    print_all_files = False

    compare_folders(folder_a, folder_b, print_all_files)
