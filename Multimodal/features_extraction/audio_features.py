"""
In this module, we extract audio features from the audio files.
We will explore and extract features using differents approaches:
1. Using OpenSMILE library to extract features such as MFCC, Chroma, Mel, Contrast, Tonnetz, and Spectral Centroid.
2. Using Wav2Vec2 model to extract features from the audio files.
3. Using Deep Spectrum to extract features from the audio files.

Then, we use pandas library to save the extracted features in a CSV file.
"""

import os
import numpy as np
import pandas as pd
import torch