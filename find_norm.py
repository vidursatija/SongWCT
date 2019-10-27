import numpy as np
import librosa
import os
from tqdm import tqdm

base_path = "cut_wavs"

all_audios = [os.path.join(base_path, f) for f in os.listdir(base_path)]

all_vals = []
xi_sq = []
means = []

for a in tqdm(all_audios):
    y, sr = librosa.load(a, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # [128, t] # this function squares the values
    S_dB = librosa.power_to_db(S, ref=np.max).reshape([-1]) # don't need to square here
    mean = np.mean(S_dB)
    means.append(mean)
    xi_sq.append(np.mean(np.power(S_dB, 2)))

mean = np.mean(means)
xi_sq_mean = np.mean(xi_sq)
stddev = np.sqrt(xi_sq_mean - mean*mean)
print(mean, stddev)
