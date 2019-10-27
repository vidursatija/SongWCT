import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("cut_wavs/part_200_10.wav", sr=16000)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
print(S)

S_dB = librosa.power_to_db(S, ref=np.max)

print(S_dB)

librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.show()