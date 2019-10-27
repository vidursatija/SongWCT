import numpy as np
import librosa
import os
import sys
from tqdm import tqdm

music_dir = "../../Music/iTunes/iTunes Media/Music"

all_singers = [os.path.join(music_dir, f) for f in os.listdir(music_dir)]

all_albums = []
for singer in tqdm(all_singers):
    try:
        all_albums.extend([os.path.join(singer, f) for f in os.listdir(singer)])
    except:
        pass

all_songs = []
for album in tqdm(all_albums):
    try:
        all_songs.extend([os.path.join(album, f) for f in os.listdir(album) if f.endswith(".mp3")])
    except:
        pass

print(len(all_songs))

segment_size = 10  # seconds
def_sr = 16000
segment_len = def_sr*segment_size

output_folder = "./cut_wavs/"

for i, song in tqdm(enumerate(all_songs)):
    y, sr = librosa.load(song, sr=def_sr)
    n = len(y)//segment_len
    for j in range(n):
        y1 = y[j*segment_len: (j+1)*segment_len]
        librosa.output.write_wav(os.path.join(output_folder,
                                 "part_%d_%d.wav" % (i, j)), y1, sr)