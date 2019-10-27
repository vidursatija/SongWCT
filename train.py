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

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--encoder', type=str, default=None,
                    help='Encoder path')
parser.add_argument('--decoder', type=str, default=None,
                    help='Decoder path')
parser.add_argument('--optimizer', type=str, default=None,
                    help='Optimizer path')
parser.add_argument('--x', type=int, default=1,
                    help='Number of AE layers to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--learn_rate', type=float, default=0.001,
                    help='Learning rate')

args = parser.parse_args()

transform = tvt.Compose(
    [transforms.AmplitudeToDB(stype='power', top_db=None), # it won't square the input
     tvt.Normalize(mean=[-38.39992], std=[13.462255])])


class MyIterableDataset(torch.utils.data.Dataset):
    def __init__(self, audios, transform):
        super(MyIterableDataset).__init__()
        self.all_audios = audios  # [os.path.join(path, f) for f in os.listdir(path)]
        self.start = 0
        self.end = len(self.all_audios)
        self.transform = transform

    def __getitem__(self, index):
        output, _ = torchaudio.load(self.all_audios[index], normalization=True)

        output = transforms.MelSpectrogram(sample_rate=16000,
                                           n_fft=400, win_length=400,
                                           hop_length=160, n_mels=128)(output)
        
        output = output[:, :, :1000] # (128, 1000)

        if self.transform is not None:
            output = self.transform(output)
            output = output.squeeze(0)
        # output = output.unsqueeze(0)
        # output.requires_grad = False

        return output.detach()

    def __len__(self):
        return self.end


ds_path = "cut_wavs"
all_wavs = [os.path.join(ds_path, f) for f in os.listdir("cut_wavs")]
total_ds_len = len(all_wavs)
train_split = int(0.9*total_ds_len)
test_split = total_ds_len - train_split
random.shuffle(all_wavs)

train_files = all_wavs[:train_split]
test_files = all_wavs[train_split:]

train_dataset = MyIterableDataset(train_files, transform)
test_dataset = MyIterableDataset(test_files, transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=12)

num_layers = args.x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

if args.encoder:
    encoder = models.encoder(x=num_layers, pretrained_path=args.encoder).to(device)
else:
    encoder = models.encoder(x=num_layers).to(device)

if args.decoder:
    decoder = models.decoder(x=num_layers, pretrained_path=args.decoder).to(device)
else:
    decoder = models.decoder(x=num_layers).to(device)

encoder.train(True)
decoder.train(True)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(list(decoder.parameters())+list(encoder.parameters()),
                       lr=args.learn_rate) # .to(device)
if args.optimizer:
    optimizer.load_state_dict(torch.load(args.optimizer))
z_loss = 0.01

step = 0
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    for data in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        try:
            inputs = data.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            z, maxpool = encoder(inputs)
            inputs_hat = decoder(z.to(device), maxpool)
            z_hat, _ = encoder(inputs_hat.to(device))
            loss = (criterion(inputs_hat, inputs) + z_loss*criterion(z_hat, z))/(1+z_loss)
            loss.backward()
            optimizer.step()
            step += 1
        except Exception as e:
            print(e)
            continue

        # print statistics
        running_loss += loss.item()
        if i % 400 == 399:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / 400))
            running_loss = 0.0
            torch.save(decoder.state_dict(), "model_"+str(num_layers)+"/dec_"+str(step)+".pkl")
            torch.save(encoder.state_dict(), "model_"+str(num_layers)+"/enc_"+str(step)+".pkl")
            torch.save(optimizer.state_dict(), "model_"+str(num_layers)+"/opt_"+str(step)+".pkl")
        i += 1

    torch.save(decoder.state_dict(), "model_"+str(num_layers)+"/dec_"+str(step)+".pkl")
    torch.save(encoder.state_dict(), "model_"+str(num_layers)+"/enc_"+str(step)+".pkl")
    torch.save(optimizer.state_dict(), "model_"+str(num_layers)+"/opt_"+str(step)+".pkl")
    i = 0
    running_loss = 0.0
    for data in tqdm(testloader):
        # get the inputs; data is a list of [inputs, labels]
        try:
            inputs = data.to(device)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            z, maxpool = encoder(inputs)
            inputs_hat = decoder(z.to(device), maxpool)
            z_hat, _ = encoder(inputs_hat.to(device))
            loss = criterion(inputs_hat, inputs) + 0.01*criterion(z_hat, z)
            # loss.backward()
            # optimizer.step()
        except Exception as e:
            print(e)
            continue
        i += 1
        # print statistics
        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / i))

print('Finished Training')
