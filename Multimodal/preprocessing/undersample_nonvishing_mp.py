import zipfile
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

VALID_CATEGORIES = {
    "교육": "ALL",
    "민원": "ALL",
    "HR": ["기업교육문의"],
    "전자상거래": ["게시판/이벤트 문의", "배송, 반송 문의"]
}

def is_valid_category(category, subcategory):
    """
    Checks if a given category and subcategory are valid based on predefined
    rules and categories.

    This function validates the main category and optionally its subcategory.
    It considers a category valid if it is either listed in the predefined
    list of acceptable categories or if it adheres to the specific subcategory
    rules defined in these categories.

    Parameters
    ----------
    category : str
        The main category to validate.
    subcategory : str
        The subcategory to validate within the main category.

    Returns
    -------
    bool
        True if the category and subcategory are valid, False otherwise.
    """
    if category in ["교육", "민원"]:
        return True
    if category in VALID_CATEGORIES:
        if VALID_CATEGORIES[category] == "ALL":
            return True
        if subcategory in VALID_CATEGORIES[category]:
            return True
    return False

def parse_jsons(zip_folder):
    """
    Parses a set of JSON files within zip archives, extracts conversation data, and organizes it into a pandas DataFrame.

    The function processes zip archives located in a specified folder. Each archive contains JSON files.
    The JSON files are parsed to extract dialog information and metadata, which is then filtered based on
    category and subcategory criteria. Valid conversations are stored in a structured DataFrame.

    Parameters:
    zip_folder: str
        The path to the directory containing the zip files named 'TL_D01.zip' to 'TL_D04.zip'.

    Returns:
    pandas.DataFrame
        A DataFrame with extracted conversation data. Each row corresponds to a conversation, including its folder
        path, the combined transcript, category, and subcategory.
    """
    conversation_dict = {}
    for i in range(1, 5):
        tl_zip_path = os.path.join(zip_folder, f'TL_D0{i}.zip')
        with zipfile.ZipFile(tl_zip_path, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            for json_file in tqdm(json_files, desc=f'Parsing {tl_zip_path}'):
                with zip_ref.open(json_file) as f:
                    data = json.load(f)
                    type_info = data['dataSet']['typeInfo']
                    category = type_info['category']
                    subcategory = type_info['subcategory']
                    if is_valid_category(category, subcategory):
                        dialogs = data['dataSet']['dialogs']
                        if dialogs:
                            conv_folder = os.path.dirname(dialogs[0]['audioPath'])
                            transcript_combined = " ".join([d['text'] for d in dialogs])
                            conversation_dict[conv_folder] = {
                                'conversation_folder': conv_folder,
                                'transcript_combined': transcript_combined,
                                'category': category,
                                'subcategory': subcategory
                            }
    return pd.DataFrame(conversation_dict.values())

def extract_single_conversation(args):
    """
    Extracts a single conversation from a set of zipped files.

    This function processes a given conversation folder along with its associated
    row data and attempts to locate and extract audio and transcript files
    from a specified set of zip archives. If the folder is found in the zip
    files, the matching audio files are extracted to the designated output location.
    The conversation transcript is also written to a text file in the output folder.
    If the folder is not found in any of the zip files, a warning is printed.

    Parameters:
        args (tuple): A tuple containing the following elements:
            folder (str): The name of the conversation folder to be extracted.
            row (dict): A dictionary containing information about the conversation,
                including a key 'transcript_combined' with the transcript text.
            zip_folder (str): Path to the directory containing the zip files.

    Returns:
        None
    """
    folder, row, zip_folder = args
    extracted = False
    for i in range(1, 5):
        ts_zip_path = os.path.join(zip_folder, f'TS_D0{i}.zip')
        with zipfile.ZipFile(ts_zip_path, 'r') as zip_ref:
            folder_path = folder + '/'
            files_to_extract = [f for f in zip_ref.namelist() if f.startswith(folder_path) and f.endswith('.wav')]
            if files_to_extract:
                for audio_file in files_to_extract:
                    target_path = os.path.join(output_folder, audio_file)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(audio_file) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                # Save transcript
                transcript_path = os.path.join(output_folder, folder, 'transcript.txt')
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(row['transcript_combined'])
                extracted = True
                break
    if not extracted:
        print(f"Warning: Folder {folder} not found in TS_D0X.zip files.")

def main():
    """
    Main function for randomly undersampling a non-vishing dataset using multiprocessing.

    This script processes a folder of zip files containing conversation datasets, filters
    valid conversations, samples a specified number of them, extracts their audio
    files in parallel using multiple workers, and generates a manifest CSV file.

    Arguments:
        --zip_folder: str
            Path to the folder containing `TS_` and `TL_` zip files. This is required
            to locate datasets for processing.
        --output_folder: str
            Path to the folder where the output balanced dataset and manifest CSV
            file will be saved.
        --sample_size: int, optional
            Number of conversation folders to extract. Defaults to 706.
        --num_workers: int, optional
            Number of parallel processes to utilize for audio file extraction.
            Defaults to 10.
    """
    parser = argparse.ArgumentParser(description='Randomly undersample non-vishing dataset with multiprocessing.')
    parser.add_argument('--zip_folder', required=True, help='Path to folder containing TS_ and TL_ zip files')
    parser.add_argument('--output_folder', required=True, help='Path to output folder for balanced dataset')
    parser.add_argument('--sample_size', type=int, default=706, help='Number of conversation folders to extract')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel processes to use')

    args = parser.parse_args()
    global output_folder
    output_folder = args.output_folder

    print("\n[Step 1] Parsing JSON files and filtering categories...")
    master_df = parse_jsons(args.zip_folder)
    print(f"Total valid conversations after filtering: {len(master_df)}")

    print(f"\n[Step 2] Randomly sampling {args.sample_size} conversation folders...")
    balanced_df = master_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    print(f"\n[Step 3] Extracting with {args.num_workers} parallel workers...")
    tasks = [(row['conversation_folder'], row, args.zip_folder) for idx, row in balanced_df.iterrows()]

    # Use multiprocessing to extract audio files
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(extract_single_conversation, tasks), total=len(tasks))) # Use imap_unordered for better performance

    print("\n[Step 4] Generating manifest CSV...")
    balanced_df['label'] = 0 # Assuming label 0 for non-vishing
    manifest_path = os.path.join(args.output_folder, 'non_vishing_manifest.csv') # Path to save the manifest
    balanced_df.to_csv(manifest_path, index=False)

    print(f"\nDone! Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()