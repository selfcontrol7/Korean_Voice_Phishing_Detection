import librosa
import numpy as np
import opensmile
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import tempfile
import subprocess
import os
import soundfile as sf

# Initialize models once
_w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
_w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_mfcc(y, sr=16000, n_mfcc=13, n_fft=400, hop_length=160):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T

def extract_egemaps(y, sr=16000):
    # Save to temp wav because opensmile expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # librosa.output.write_wav(tmp.name, y, sr)
        sf.write(tmp.name, y, sr)
        feats = smile.process_file(tmp.name)
    os.remove(tmp.name)
    return feats.values

def extract_wav2vec2(y, sr=16000):
    inputs = _w2v_processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _w2v_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()

def extract_deepspectrum(y, sr=16000, config_path="config/deepspec.yaml", output_dir="features/deepspectrum_feats"):
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # librosa.output.write_wav(tmp.name, y, sr)
        sf.write(tmp.name, y, sr)
        base = os.path.splitext(os.path.basename(tmp.name))[0]
        out_csv = os.path.join(output_dir, f"{base}_deepspec.csv")
        cmd = ["deepspectrum", "-m", config_path, "-i", tmp.name, "-o", out_csv]
        subprocess.run(cmd, check=True)
    feats = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    os.remove(tmp.name)
    os.remove(out_csv)
    return feats
