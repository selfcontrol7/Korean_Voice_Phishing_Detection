# features_extraction/build_features.py

import os
import io
import tempfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
from pydub import AudioSegment
from pydub.utils import which

# Specify the full path to ffprobe.exe (update the path accordingly)
# AudioSegment.ffprobe = r'C:ffmpeg\bin\ffprobe.exe'
# AudioSegment.converter = r'C:ffmpeg\bin\ffmpeg.exe'

# point pydub at your ffmpeg/ffprobe
# AudioSegment.converter = which("C:ffmpeg/bin/ffmpeg.exe")
# AudioSegment.ffprobe   = which("C:ffmpeg/bin/ffprobe.exe")

# import the four feature‐extraction functions
from .audio_features import (
    extract_mfcc,
    extract_egemaps,
    extract_deepspectrum,
    extract_wav2vec2
)

# root of the repo
ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
FEATURES_DIR = ROOT / "features"

# make sure feature subfolders exist
for feat in ("mfcc", "egemaps", "wav2vec2", "deep_spectrum"):
    (FEATURES_DIR / feat).mkdir(parents=True, exist_ok=True)


def prepare_audio(infile: Path, sr: int = 16000) -> Path:
    """
    Load any mp3/mp4/wav/… into a 16 kHz mono WAV file,
    return a Path to the temporary WAV.
    """
    # check if the file exists
    if not infile.exists():
        raise FileNotFoundError(f"File {infile} does not exist.")
    # check if the file is empty
    if infile.stat().st_size == 0:
        raise ValueError(f"File {infile} is empty.")
    # check if the file is a valid audio file
    if not infile.suffix in [".mp3", ".mp4", ".wav", ".flac"]:
        raise ValueError(f"File {infile} is not a valid audio file.")

    sound = AudioSegment.from_file(str(infile))
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(sr)

    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    sound.export(str(tmp_wav), format="wav")
    return tmp_wav


def process_file(audio_path: Path, label: int):
    call_id = audio_path.stem  # filename without extension

    # 1) normalize to WAV
    wav_path = prepare_audio(audio_path)
    # wav_path = audio_path
    print(f"Processing {wav_path}...")

    # 2) load & extract
    mfcc_feats = extract_mfcc(str(wav_path))
    egemaps_feats = extract_egemaps(str(wav_path))
    wav2vec_feats = extract_wav2vec2(str(wav_path))
    # ds_feats = extract_deepspectrum(str(wav_path))

    # 3) save out
    np.save(FEATURES_DIR / "mfcc"        / f"{call_id}.npy", mfcc_feats)
    np.save(FEATURES_DIR / "egemaps"     / f"{call_id}.npy", egemaps_feats)
    np.save(FEATURES_DIR / "wav2vec2"    / f"{call_id}.npy", wav2vec_feats)
    # np.save(FEATURES_DIR / "deep_spectrum"/ f"{call_id}.npy", ds_feats)

    # optionally return metadata for a manifest
    return {
        "call_id": call_id,
        "label": label,
        "mfcc":    str(FEATURES_DIR / "mfcc"         / f"{call_id}.npy"),
        "egemaps": str(FEATURES_DIR / "egemaps"      / f"{call_id}.npy"),
        "wav2vec": str(FEATURES_DIR / "wav2vec2"     / f"{call_id}.npy"),
        # "ds":      str(FEATURES_DIR / "deep_spectrum"/ f"{call_id}.npy"),
    }


def main():
    metas = []
    failed = []
    for cls, subdir in [("vishing", 1), ("non_vishing", 0)]:
        folder = AUDIO_DIR / cls
        # gather all audio files recursively
        files = list(folder.rglob("*.*"))
        print(f"Processing {len(files)} audio files...")
        print("=" * 250)

        for f in tqdm(files[:5], desc=f"→ {cls}"):
            # print("working on file:", f)
            # check if the file exists
            if not f.exists():
                print(f"File {f} does not exist.")
                continue
            try:
                meta = process_file(f, subdir)
                metas.append(meta)
            except Exception as e:
                tqdm.write(f"[!] failed on {f}: {e}")
                # add the failed file to the list of failed files
                failed.append({
                    "file": str(f),
                    "error": str(e)
                })

    # Print the number of files processed
    print(f"Processed {len(metas)} audio files.")
    print("=" * 250)

    # Print the first few entries in the metadata
    print("Sample metadata entries:")
    for meta in metas[:3]:
        print(meta)
    print("=" * 250)

    # Print the number of failed files
    print(f"Number of failed files: {len(failed)}")
    # Print the failed files
    print("Failed files:")
    for fail in failed:
        print(fail)
    print("=" * 300)


    # # If you want, dump a small JSON manifest of paths+labels
    # import json
    # with open(ROOT / "data" / "features_manifest.json", "w", encoding="utf8") as out:
    #     json.dump(metas, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
