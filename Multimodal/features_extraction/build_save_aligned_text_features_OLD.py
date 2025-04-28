# build_save_aligned_text_features_OLD.py

import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .text_features_OLD import extract_kobert_features, extract_sbert_features

FEATURES_DIR = Path("../features")  # Directory to save features
FEATURES_DIR.mkdir(exist_ok=True)  # Create the directory if it doesn't exist


def process_segment(args):
    """
    Processes a single transcript segment to extract and save text features.

    This function handles the extraction of text features for a given transcript
    segment. The processed features include KoBERT and SR-SBERT embeddings,
    which are stored as numpy files. The segment data and its corresponding
    metadata, such as the segment ID, start and end times, and transcription text,
    are included in the output.

    Parameters:
        args (tuple): A tuple containing the following elements:
            seg_info (tuple): Metadata of the transcript segment, including the segment
                index and the segment's dictionary containing start and end times
                (in milliseconds) and transcription text.
            call_id (str): A unique identifier for the call session to which
                the segment belongs.
            label (str): The label or category assigned to the transcript segment.
            overwrite (bool): Indicates whether to overwrite existing feature files.

    Returns:
        dict: A dictionary containing the following elements:
            call_id (str): The identifier of the call session.
            segment_id (str): The unique ID formed by combining the call ID and
                segment index.
            start (float): The start time of the segment in seconds.
            end (float): The end time of the segment in seconds.
            text (str): The transcription text of the segment.
            label (str): The label assigned to the segment.
            kobert_path (str): The file path where the KoBERT features are saved.
            sbert_path (str): The file path where the SR-SBERT features are saved.
    """
    seg_info, call_id, label, overwrite = args  # Unpack the arguments
    idx, seg = seg_info  # Unpack the segment information

    start_sec = seg["start"] / 1000  # Convert start time from milliseconds to seconds
    end_sec = seg["end"] / 1000
    text = seg["text"]  # Transcription text of the segment

    seg_id = f"{call_id}_{idx}"

    paths = {}  # Dictionary to store paths of the saved features

    # KoBERT
    kobert_path = FEATURES_DIR / "kobert" / f"{seg_id}.npy"
    if overwrite or not kobert_path.exists():
        kobert_feats = extract_kobert_features(text)
        np.save(kobert_path, kobert_feats)
    paths["kobert_path"] = str(kobert_path)

    # SR-SBERT
    sbert_path = FEATURES_DIR / "sbert" / f"{seg_id}.npy"
    if overwrite or not sbert_path.exists():
        sbert_feats = extract_sbert_features(text)
        np.save(sbert_path, sbert_feats)
    paths["sbert_path"] = str(sbert_path)

    # Return the segment information and paths to the saved features
    return {
        "call_id": call_id,
        "segment_id": seg_id,
        "start": start_sec,
        "end": end_sec,
        "text": text,
        "label": label,
        **paths
    }


# Main function to handle command line arguments and process transcript segments
def main():
    parser = argparse.ArgumentParser()  # Create an argument parser
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)  # Split to process
    parser.add_argument("--overwrite", action="store_true")  # Flag to overwrite existing features
    parser.add_argument("--workers", type=int, default=4)  # Number of parallel workers
    args = parser.parse_args()  # Parse the command line arguments

    # Create directories for saving features if they don't exist
    for feat in ["kobert", "sbert"]:
        (FEATURES_DIR / feat).mkdir(exist_ok=True)

    manifest_path = Path(f"data/{args.split}_manifest.jsonl")  # Path to the manifest file
    with open(manifest_path, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    # Check if the manifest file is empty
    if not entries:
        print(f"Manifest file {manifest_path} is empty or does not exist.")
        return
    else:
        print(f"Manifest file {manifest_path} loaded with {len(entries)} entries.")
    print("=" * 100)

    # Initialize a list to store the segment information
    segment_manifest = []  # List to store the segment information

    # Process each entry in the manifest file
    for entry in tqdm(entries, desc=f"Processing {args.split} calls"):
        call_id = entry["call_id"]
        transcript_path = Path(entry["transcript_filepath"])
        label = entry["label"]
        print("Processing call:", call_id)
        print("Transcript path:", transcript_path)
        print("Label:", label)

        # Check if the transcript file exists
        if not transcript_path.exists():
            print(f"Transcript file {transcript_path} does not exist.")
            print("=" * 100)
            continue

        with open(transcript_path, encoding='utf-8') as f_json:
            transcript = json.load(f_json)

        segments = list(enumerate(transcript["segments"]))  # List the segments in the transcript

        with Pool(args.workers) as pool:
            results = pool.map(process_segment, [(seg, call_id, label, args.overwrite) for seg in segments])
            segment_manifest.extend(results)

    out_manifest = Path(f"data/{args.split}_text_segment_manifest.jsonl")  # Path to save the segment manifest
    with open(out_manifest, "w", encoding='utf-8') as fout:
        for item in segment_manifest:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(segment_manifest)} segments to {out_manifest}")


if __name__ == "__main__":
    main()