"""
In this module, we extract audio features from the audio files.
We will explore and extract features using differents approaches:
1. Using OpenSMILE library to extract features such as MFCC, Chroma, Mel, Contrast, Tonnetz, and Spectral Centroid.
2. Using Wav2Vec2 model to extract features from the audio files.
3. Using Deep Spectrum to extract features from the audio files.

Then, we use pandas library to save the extracted features in a CSV file.
"""

import librosa
import numpy as np
import opensmile
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import subprocess
import os

# Load once to avoid reloading model every call
_w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
_w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_mfcc(audio_path, sr=16000, n_mfcc=13, n_fft=400, hop_lenght=160):
    """
    Extract MFCC features from audio file.

    :param audio_path: Path to the audio file
    :param sr: Sample rate
    :param n_mfcc: Number of MFCC features to extract
    :param n_fft: FFT window size
    :param hop_lenght: Hop length

    :return: MFCC features
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_lenght)
    print("MFCC features extracted with shape:", mfcc.shape)
    return mfcc.T

def extract_egemaps(audio_path):
    """
    Extract eGeMAPS features from an audio file using OpenSMILE.

    :param audio_path: Path to the audio file

    :return: pandas.DataFrame of eGeMAPS features functionals
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(audio_path)
    print("eGeMAPS features extracted with shape:", features.shape)
    return features


def extract_wav2vec2(audio_path, sr=16000):
    """
    Extract wav2vec2 embeddings (last hidden state) for an audio file.

    :param audio_path: Path to the audio file
    :param sr: Sample rate

    :return: Wav2Vec2 features as np.ndarray of shape (num_frames, hidden_size)
    """

    # Load and preprocess the audio file
    audio_input, _ = librosa.load(audio_path, sr=sr)
    inputs = _w2v_processor(audio_input, sampling_rate=sr, return_tensors="pt", padding=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = _w2v_model(**inputs)
        # (1, seq_len, hidden_size)
    last_hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # Return the last hidden states as numpy array
    pritn("Wav2Vec2 features extracted with shape:", last_hidden_states.shape)
    return last_hidden_states

def extract_deepspectrum(audio_path, config_path, output_dir="../features/deep_spectrum/DeepSpectrum_output.csv"):
    """
    ="DeepSpectrum/DeepSpectrum.h5"
    Extract DeepSpectrum features using the DeepSpectrum CLI.

    :param audio_path: Path to the audio file
    :param config_path: Path to the Deep Spectrum configuration file
    :param output_dir: Path to save the output features

    :return: np.ndarray of shape (num_frames, num_features)
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_csv = os.path.join(output_dir, f"{base}_deepspec.csv")

    # Check if DeepSpectrum is installed
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DeepSpectrum configuration file {config_path} not found.")

    # Run Deep Spectrum command
    # cmd = f"python DeepSpectrum/deep_spectrum.py --model {config_path} --input {audio_path} --output {output_dir}"
    # subprocess.run(cmd, shell=True)

    cmd = [
        "deep_spectrum",
        "-m", config_path,
        "-i", audio_path,
        "-o", out_csv
    ]
    subprocess.run(cmd, check=True)
    feats = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    print("DeepSpectrum features extracted with shape:", feats.shape)
    return feats
