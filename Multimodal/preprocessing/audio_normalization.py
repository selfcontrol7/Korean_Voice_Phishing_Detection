# """
# Prepare the audio file for Wav2Vec2 processing
# """
# import tempfile
# from pathlib import Path
# from tempfile import NamedTemporaryFile
#
# import librosa
# import numpy as np
# from pydub import AudioSegment
# import soundfile as sf
# from pydub.utils import which
# from tqdm import tqdm
#
# # point pydub at your ffmpeg/ffprobe
# # AudioSegment.converter = which("C:ffmpeg/bin/ffmpeg.exe")
# # AudioSegment.ffprobe   = which("C:ffmpeg/bin/ffprobe.exe")
#
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
#
#
# # # def prepare_audio(audio_path: str, target_sr: int =16000) -> str:
# # def prepare_audio(infile: Path, sr: int = 16000) -> Path:
# #     """
# #     Load any audio file (mp3/mp4/wav/etc), convert to mono WAV @ target_sr,
# #     and write to a temporary file. Returns the temp file path.
# #
# #     :param audio_path: Path to the input audio file
# #     :param target_sr: Target sample rate
# #
# #     :return: Path to the processed audio file1
# #     """
# #     # Load the audio file using pydub, Decode MP3/MP4/WAV → pydub.AudioSegment
# #     # audio = AudioSegment.from_file(audio_path)
# #     audio = AudioSegment.from_file(str(infile))
# #
# #     # Convert to mono and set the sample rate to target_sr (16 kHz)
# #     audio = audio.set_channels(1).set_frame_rate(sr)
# #
# #     # save the audio to a temporary file
# #     # tmp_wav = NamedTemporaryFile(suffix='.wav', delete=False)
# #     tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
# #     audio.export(tmp_wav.name, format='wav')
# #     return tmp_wav.name
#
#
# def prepare_audio(infile: Path, sr: int = 16000) -> Path:
#     """
#     Load any mp3/mp4/wav/… into a 16 kHz mono WAV file,
#     return a Path to the temporary WAV.
#     """
#     # check if the file exists
#     if not infile.exists():
#         raise FileNotFoundError(f"File {infile} does not exist.")
#     # check if the file is empty
#     if infile.stat().st_size == 0:
#         raise ValueError(f"File {infile} is empty.")
#     # check if the file is a valid audio file
#     if not infile.suffix in [".mp3", ".mp4", ".wav", ".flac"]:
#         raise ValueError(f"File {infile} is not a valid audio file.")
#
#     sound = AudioSegment.from_file(str(infile))
#     sound = sound.set_channels(1)
#     sound = sound.set_frame_rate(sr)
#
#     tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
#     sound.export(str(tmp_wav), format="wav")
#     return tmp_wav
#
#
#
# def normalize_audio(input_path, output_path, target_sr=16000):
#     # Load the audio as a mono signal at the target sample rate
#     y, sr = librosa.load(input_path, sr=target_sr, mono=True)
#
#     # Normalize so that the maximum absolute amplitude is 1.0
#     if np.max(np.abs(y)) > 0:
#         y_normalized = y / np.max(np.abs(y))
#     else:
#         y_normalized = y
#
#     # Write the normalized audio to a new file
#     sf.write(output_path, y_normalized, sr)
#
# if __name__ == "__main__":
#     # Loop through the subdirectories in data/audio and process all the audio files
#     audio_dir = Path("../data/audio")
#     for subdir in audio_dir.iterdir():
#         if subdir.is_dir():
#             for audio_file in tqdm(subdir.iterdir(), desc=f"Processing {subdir.name}"):
#                 if audio_file.suffix in [".mp3", ".mp4", ".wav", ".flac"]:
#                     print(f"Processing {audio_file}...")
#                     # Process the audio file
#                     prepare_audio(audio_file)
#                     # Normalize the audio file
#                     # normalize_audio(audio_file, audio_file.name)
#                     print(f"Processed {audio_file} to {audio_file.name}")

# !/usr/bin/env python3
import argparse
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

VALID_EXTS = {".mp3", ".mp4", ".wav", ".flac"}


def prepare_audio_file(infile: Path, outfile: Path, sr: int = 16000):
    """
    Load a supported audio file, convert it into a mono WAV file at a sample rate of `sr`,
    and export it to `outfile`.
    """
    # Sanity checks
    if not infile.exists():
        raise FileNotFoundError(f"File {infile} does not exist.")
    if infile.stat().st_size == 0:
        raise ValueError(f"File {infile} is empty.")
    if infile.suffix.lower() not in VALID_EXTS:
        raise ValueError(f"File {infile} is not a valid audio file (ext={infile.suffix}).")

    # Load the file and perform conversion
    sound = AudioSegment.from_file(str(infile))
    sound = sound.set_channels(1).set_frame_rate(sr)

    # Ensure destination folder exists
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Export as WAV
    sound.export(str(outfile), format="wav")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert all audio under --src to 16kHz mono WAV files under --dest"
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source folder (e.g. data/audio/vishing)"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination folder (e.g. data/audio/vishing/wav)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sampling rate (default: 16000)"
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"{args.src} is not a directory")

    print(f"Processing all audio in {args.src} → {args.dest} at {args.sr} Hz…")
    for ext in VALID_EXTS:
        for infile in tqdm(list(args.src.glob(f"*{ext}")), desc=f"Processing {ext} files"):
            # Compute a relative path under src and map it to the destination folder
            rel_path = infile.relative_to(args.src)
            outfile = args.dest / rel_path.with_suffix(".wav")
            tqdm.write(f"  Converting {infile} → {outfile}")
            prepare_audio_file(infile, outfile, sr=args.sr)
    print("All done.")


if __name__ == "__main__":
    main()