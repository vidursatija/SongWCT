import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio
import torchaudio.transforms as transforms
import torchvision.transforms as tvt
import torch.optim as optim

import models
from tqdm import tqdm
import os
import random
import math
import librosa

import argparse
import mat_transforms

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--encoder', type=str, default=None,
                    help='Encoder path')
parser.add_argument('--decoder', type=str, default=None,
                    help='Decoder path')
parser.add_argument('--style', type=str, default=None,
                    help='Style image path')
parser.add_argument('--content', type=str, default=None,
                    help='Content image path')
parser.add_argument('--x', type=int, default=1,
                    help='Number of AE layers to use')

args = parser.parse_args()


def load_audio(fname, transform):
    # output, _ = torchaudio.load(fname, normalization=True)

    # output = transforms.MelSpectrogram(sample_rate=16000,
    #                                    n_fft=400, win_length=400,
    #                                    hop_length=160, n_mels=128)(output)
    y, _ = librosa.load(fname, 16000)
    stft_y = librosa.core.stft(y, n_fft=400, hop_length=160, win_length=400)
    mag = np.abs(stft_y)
    ang = np.angle(stft_y)
    D = mag**2
    output = librosa.feature.melspectrogram(S=D, sr=16000)

    output = torch.Tensor(output)
    output = output.unsqueeze(0)

    output = output[:, :, :1000]
    ang = ang[:, :1000]
    if transform is not None:
        output = transform(output)
    
    # output = output.unsqueeze(0)  # [1, 128, 1000]
    return output, ang


transform = tvt.Compose(
        [transforms.AmplitudeToDB(stype='power', top_db=None),
         tvt.Normalize(mean=[-38.39992], std=[13.462255])])

reverse_normalize = tvt.Normalize(
    mean=[38.39992/13.462255],
    std=[1./13.462255])

num_layers = args.x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

encoder = models.encoder(x=num_layers, pretrained_path=args.encoder).to(device)
decoder = models.decoder(x=num_layers, pretrained_path=args.decoder).to(device)

encoder.train(False)
decoder.train(False)

content_audio, content_ang = load_audio(args.content, transform)
content_audio = content_audio.to(device)
style_audio, _ = load_audio(args.style, transform)
style_audio = style_audio.to(device)

z_content, maxpool_content = encoder(content_audio)  # (1, C, W)
z_style, _ = encoder(style_audio)  # (1, C, W)

n_channels = z_content.size()[1]  # C
n_1 = z_content.size()[2]  # W

z_content = z_content.squeeze(0) # .view([n_channels, -1])  # (C, HW)
z_style = z_style.squeeze(0) # .view([n_channels, -1])  # (C, HW)

white_content = mat_transforms.whitening(z_content.cpu().detach().numpy())  # (C, HW)
color_content = mat_transforms.colouring(z_style.cpu().detach().numpy(), white_content)  # (C, HW)

# alpha = 0.6
# color_content = alpha*color_content + (1.-alpha)*z_content.cpu().detach().numpy()

color_content = torch.Tensor(color_content) # tvt.ToTensor()(color_content)
# color_content = color_content.view([n_channels, n_1, n_2]) # (C, H, W)
color_content = color_content.unsqueeze(0) # (1, C, W)

inputs_hat = decoder(color_content.to(device), maxpool_content)

new_audio = inputs_hat # .squeeze(0) # (1, C, W)
new_audio = reverse_normalize(new_audio) # (1, C, W)
new_audio = new_audio[0] # take only 1 channel (C, W)

new_audio = np.maximum(np.minimum(new_audio.cpu().detach().numpy(), 0.0), -80.0)
power_spectro = librosa.core.db_to_power(new_audio, ref=1.0)

inv_power = librosa.feature.inverse.mel_to_stft(power_spectro, sr=16000, n_fft=400, power=2.0)

full_stft = inv_power*np.exp(1j*content_ang)
y_new = librosa.core.istft(full_stft, hop_length=160, win_length=400)
# y_new = librosa.feature.inverse.mel_to_audio(power_spectro, sr=16000,
#                                              n_fft=400, hop_length=160,
#                                              win_length=400, n_iter=32) # it will take sqrt

librosa.output.write_wav("processed.wav", y_new, 16000)
