import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
import os
from glob import glob
import librosa
from torchvision.models import resnet34

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav = np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
        spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def lr_decay(optimizer, epoch):
    if epoch%20==0:
        new_lr = learning_rate / (10**(epoch//20))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer


class Data(Dataset):
    def __init__(self, wavs_path):
       
        self.data = []
        self.labels = []
        self.wavs_path = wavs_path
        self.n_fft= int(4096)
        self.hop_length= int(self.n_fft/4) #
        self.top_db = 80
        self.fmin = 20
        self.fmax = 8300
        self.sr = int(22050 * 1.0)
       
        
        for path in tqdm(self.wavs_path):
            self.labels.append(int(path.split('/')[-2]))
            
            wav, sr = librosa.load(path)
            start_idx = 0
            for i in range(wav.shape[0]):
                if abs(wav[i]) < 0.025: continue
                start_idx = i
                break
            wav_cut = wav[start_idx:start_idx + int(self.sr)]
            shape = wav_cut.shape[0]
            if  shape< self.sr:
                wav_cut = np.pad(wav_cut,int(np.ceil((1* self.sr-shape)/2)),mode='constant')
                wav_cut = wav_cut[: self.sr]
            

            if wav_cut.shape[0] !=  self.sr:
                print(path, wav_cut.shape)
                
            spec=librosa.feature.melspectrogram(wav_cut, sr=self.sr, n_fft=self.n_fft,hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax)
            spec_db=librosa.power_to_db(spec,top_db=self.top_db)
            self.data.append(spec_to_image(spec_db))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')


resnet_model = resnet34(pretrained = True)
resnet_model.fc = nn.Linear(512, 20)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet_model = resnet_model.to(device)

wavs_path = glob(os.path.join('/home/lab/Documents/Human/Elevator_Sound_Classification/Record', '*', '*'))
train_data = Data(wavs_path)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 2e-5
optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
epochs = 60
train_losses = []

def train(model, loss_fn, train_loader, epochs, optimizer, train_losses, change_lr=None):
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        batch_losses=[]
        correct = 0
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in enumerate(train_loader):
            
            x, y = data
          
            batch, height, width = x.size()
            x = x.view(batch, 1, height, width)
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
          
            for b in range(batch):
                if y[b] == (torch.argmax(y_hat[b]) +1):
                    correct+=1
            loss = loss_fn(y_hat, y-1)
            
            model.zero_grad()
            loss.backward()
            
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        percent = 100. * correct / len(train_data)
        print("epoch: {}, correct: {}/{} ({:.0f}%)".format(epoch, correct, len(train_data), percent))


train(resnet_model, loss_fn, train_loader, epochs, optimizer, train_losses, lr_decay)

wavs_path = glob(os.path.join('/home/lab/Documents/Human/Elevator_Sound_Classification/Test', '*', '*'))
test_data = Data(wavs_path)

test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

resnet_model.eval()
count = 0
for i, data in enumerate(test_loader):
    x, y = data
    batch, height, width = x.size()
    x = x.view(batch, 1, height, width)
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.long)
    y_hat = resnet_model(x)
    
    

    for b in range(batch):
        pred = (torch.argmax(y_hat[b]) +1).item()
        ground_truth = y[b].item()  
        if ground_truth!= pred:
            print("ground truth:",ground_truth, 'prediction:',pred )
            count +=1
            
print("error count", count)





      
          
        