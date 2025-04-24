import zipfile
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm

VALID_CATEGORIES = {
    "교육": "ALL",
    "민원": "ALL",
    "HR": ["기업교육문의"],
    "전자상거래": ["게시판/이벤트 문의", "배송, 반송 문의"]
}


def is_valid_category(category, subcategory):
    if category in ["교육", "민원"]:
        return True
    if category in VALID_CATEGORIES:
        if VALID_CATEGORIES[category] == "ALL":
            return True
        if subcategory in VALID_CATEGORIES[category]:
            return True
    return False


def parse_jsons(zip_folder):
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


def extract_conversation_folders(df, zip_folder, output_folder):
    for i in range(1, 5):
        ts_zip_path = os.path.join(zip_folder, f'TS_D0{i}.zip')
        with zipfile.ZipFile(ts_zip_path, 'r') as zip_ref:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Extracting conversations from TS_D0{i}.zip'):
                folder = row['conversation_folder'] + '/'
                files_to_extract = [f for f in zip_ref.namelist() if f.startswith(folder) and f.endswith('.wav')]
                for audio_file in files_to_extract:
                    target_path = os.path.join(output_folder, audio_file)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zip_ref.open(audio_file) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                # Save combined transcript
                transcript_path = os.path.join(output_folder, folder, 'transcript.txt')
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(row['transcript_combined'])


def main():
    parser = argparse.ArgumentParser(description='Randomly undersample non-vishing dataset by conversation folders.')
    parser.add_argument('--zip_folder', required=True, help='Path to folder containing TS_ and TL_ zip files')
    parser.add_argument('--output_folder', required=True, help='Path to output folder for balanced dataset')
    parser.add_argument('--sample_size', type=int, default=706, help='Number of conversation folders to extract')

    args = parser.parse_args()

    print("\n[Step 1] Parsing JSON files and filtering categories...")
    master_df = parse_jsons(args.zip_folder)
    print(f"Total valid conversations after filtering: {len(master_df)}")

    print(f"\n[Step 2] Randomly sampling {args.sample_size} conversation folders...")
    balanced_df = master_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    print("\n[Step 3] Extracting full conversation folders with transcripts...")
    extract_conversation_folders(balanced_df, args.zip_folder, args.output_folder)

    print("\n[Step 4] Generating manifest CSV...")
    balanced_df['label'] = 0
    manifest_path = os.path.join(args.output_folder, 'non_vishing_manifest.csv')
    balanced_df.to_csv(manifest_path, index=False)

    print(f"\nDone! Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
