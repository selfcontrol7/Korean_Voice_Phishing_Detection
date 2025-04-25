# build_save_aligned_audio_features.py

import json
import librosa
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from .audio_features import extract_mfcc, extract_egemaps, extract_wav2vec2, extract_deepspectrum

FEATURES_DIR = Path("features") # Directory to save features
FEATURES_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist


def process_segment(args):
    """
    Processes a single audio segment to extract and save various audio features.

    This function handles the extraction of audio features for a given audio
    segment specified by its start and end times. The processed features include
    MFCC, eGeMAPS, Wav2Vec2, and DeepSpectrum, which are stored as numpy files.
    The audio segment data and its corresponding metadata, such as the segment
    ID, start and end times, and transcription text, are included in the output.

    Parameters:
        args (tuple): A tuple containing the following elements:
            seg_info (tuple): Metadata of the audio segment, including the segment
                index and the segment's dictionary containing start and end times
                (in milliseconds) and transcription text.
            y (numpy.ndarray): The audio signal as a waveform array.
            sr (int): The sampling rate of the audio signal.
            call_id (str): A unique identifier for the audio call session to which
                the segment belongs.
            label (str): The label or category assigned to the audio segment.
            overwrite (bool): Indicates whether to overwrite existing feature files.

    Returns:
        dict: A dictionary containing the following elements:
            call_id (str): The identifier of the call session.
            segment_id (str): The unique ID formed by combining the call ID and
                segment index.
            start (float): The start time of the segment in seconds.
            end (float): The end time of the segment in seconds.
            text (str): The transcription text of the audio segment.
            label (str): The label assigned to the segment.
            mfcc_path (str): The file path where the MFCC features are saved.
            egemaps_path (str): The file path where the eGeMAPS features are saved.
            wav2vec2_path (str): The file path where the Wav2Vec2 features are saved.
            deepspectrum_path (str): The file path where the DeepSpectrum features
                are saved.
    """
    seg_info, y, sr, call_id, label, overwrite = args # Unpack the arguments
    idx, seg = seg_info # Unpack the segment information

    start_sec = seg["start"] / 1000 # Convert start time from milliseconds to seconds
    end_sec = seg["end"] / 1000
    text = seg["text"] # Transcription text of the segment

    # split the audio signal into the segment based on start and end times
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    y_seg = y[start_sample:end_sample]

    seg_id = f"{call_id}_{idx}"

    paths = {} # Dictionary to store paths of the saved features

    # MFCC
    mfcc_path = FEATURES_DIR / "mfcc" / f"{seg_id}.npy"
    if overwrite or not mfcc_path.exists(): # Check if the MFCC file already exists
        mfcc_feats = extract_mfcc(y_seg, sr)
        np.save(mfcc_path, mfcc_feats)
    paths["mfcc_path"] = str(mfcc_path)

    # eGeMAPS
    egemaps_path = FEATURES_DIR / "egemaps" / f"{seg_id}.npy"
    if overwrite or not egemaps_path.exists():
        egemaps_feats = extract_egemaps(y_seg, sr)
        np.save(egemaps_path, egemaps_feats)
    paths["egemaps_path"] = str(egemaps_path)

    # Wav2Vec2
    w2v_path = FEATURES_DIR / "wav2vec2" / f"{seg_id}.npy"
    if overwrite or not w2v_path.exists():
        w2v_feats = extract_wav2vec2(y_seg, sr)
        np.save(w2v_path, w2v_feats)
    paths["wav2vec2_path"] = str(w2v_path)

    # # DeepSpectrum
    # ds_path = FEATURES_DIR / "deepspectrum" / f"{seg_id}.npy"
    # if overwrite or not ds_path.exists():
    #     ds_feats = extract_deepspectrum(y_seg, sr)
    #     np.save(ds_path, ds_feats)
    # paths["deepspectrum_path"] = str(ds_path)

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

# Main function to handle command line arguments and process audio segments
def main():
    parser = argparse.ArgumentParser() # Create an argument parser
    parser.add_argument("--split", choices=["train", "val", "test"], required=True) # Split to process
    parser.add_argument("--overwrite", action="store_true") # Flag to overwrite existing features
    parser.add_argument("--workers", type=int, default=4) # Number of parallel workers
    args = parser.parse_args() # Parse the command line arguments

    # Create directories for saving features if they don't exist
    for feat in ["mfcc", "egemaps", "wav2vec2", "deepspectrum"]:
        (FEATURES_DIR / feat).mkdir(exist_ok=True)

    manifest_path = Path(f"data/{args.split}_manifest.jsonl") # Path to the manifest file
    with open(manifest_path, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    # Check if the manifest file is empty
    if not entries:
        print(f"Manifest file {manifest_path} is empty or does not exist.")
        return
    else:
        print(f"Manifest file {manifest_path} loaded with {len(entries)} entries.")
    print("=" * 250)

    # Initialize a list to store the segment information
    segment_manifest = [] # List to store the segment information

    #Process each entry in the manifest file
    for entry in tqdm(entries, desc=f"Processing {args.split} calls"):
        call_id = entry["call_id"]
        audio_path = Path(entry["audio_filepath"])
        transcript_path = Path(entry["transcript_filepath"])
        label = entry["label"]
        print("Processing call:", call_id)
        # print("Audio path:", audio_path)
        # print("Transcript path:", transcript_path)
        # print("Label:", label)

        # Check if the audio and transcript files exist
        if not audio_path.exists():
            print(f"Audio file {audio_path} does not exist.")
            print("=" * 250)
            continue
        if not transcript_path.exists():
            print(f"Transcript file {transcript_path} does not exist.")
            print("=" * 250)
            continue

        y, sr = librosa.load(audio_path, sr=16000) # Load the audio file
        with open(transcript_path, encoding='utf-8') as f_json:
            transcript = json.load(f_json)

        segments = list(enumerate(transcript["segments"])) # List the segments in the transcript

        with Pool(args.workers) as pool:
            results = pool.map(process_segment, [(seg, y, sr, call_id, label, args.overwrite) for seg in segments])
            segment_manifest.extend(results)

    out_manifest = Path(f"data/{args.split}_segment_manifest.jsonl") # Path to save the segment manifest
    with open(out_manifest, "w", encoding='utf-8') as fout:
        for item in segment_manifest:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(segment_manifest)} segments to {out_manifest}")


if __name__ == "__main__":
    main()
