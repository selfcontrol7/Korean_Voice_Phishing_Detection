"""
Prepare the audio file for Wav2Vec2 processing
"""
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from pydub.utils import which
from tqdm import tqdm

# point pydub at your ffmpeg/ffprobe
# AudioSegment.converter = which("C:ffmpeg/bin/ffmpeg.exe")
# AudioSegment.ffprobe   = which("C:ffmpeg/bin/ffprobe.exe")

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"


# # def prepare_audio(audio_path: str, target_sr: int =16000) -> str:
# def prepare_audio(infile: Path, sr: int = 16000) -> Path:
#     """
#     Load any audio file (mp3/mp4/wav/etc), convert to mono WAV @ target_sr,
#     and write to a temporary file. Returns the temp file path.
#
#     :param audio_path: Path to the input audio file
#     :param target_sr: Target sample rate
#
#     :return: Path to the processed audio file1
#     """
#     # Load the audio file using pydub, Decode MP3/MP4/WAV → pydub.AudioSegment
#     # audio = AudioSegment.from_file(audio_path)
#     audio = AudioSegment.from_file(str(infile))
#
#     # Convert to mono and set the sample rate to target_sr (16 kHz)
#     audio = audio.set_channels(1).set_frame_rate(sr)
#
#     # save the audio to a temporary file
#     # tmp_wav = NamedTemporaryFile(suffix='.wav', delete=False)
#     tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
#     audio.export(tmp_wav.name, format='wav')
#     return tmp_wav.name


def prepare_audio(infile: Path, sr: int = 16000):
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

    # tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    # sound.export(str(tmp_wav), format="wav")

    # save the audio files to the folderdata
    tmp_wav = Path("../data") / infile.name
    sound.export(str(tmp_wav), format="wav")


def normalize_audio(input_path, output_path, target_sr=16000):
    # Load the audio as a mono signal at the target sample rate
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)

    # Normalize so that the maximum absolute amplitude is 1.0
    if np.max(np.abs(y)) > 0:
        y_normalized = y / np.max(np.abs(y))
    else:
        y_normalized = y

    # Write the normalized audio to a new file
    sf.write(output_path, y_normalized, sr)

if __name__ == "__main__":
    # Loop through the subdirectories in data/audio and process all the audio files
    audio_dir = Path("../data/audio")
    for subdir in audio_dir.iterdir():
        if subdir.is_dir():
            for audio_file in tqdm(subdir.iterdir(), desc=f"Processing {subdir.name}"):
                if audio_file.suffix in [".mp3", ".mp4", ".wav", ".flac"]:
                    print(f"Processing {audio_file}...")
                    # Process the audio file
                    # prepare_audio(audio_file)
                    # Normalize the audio file
                    normalize_audio(audio_file, audio_file.name)
                    print(f"Processed {audio_file} to {audio_file.name}")
