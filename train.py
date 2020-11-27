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



class Data(Dataset):
    def __init__(self, wavs_path):
       
        self.data = []
        self.labels = []
        self.wavs_path = wavs_path
        self.n_fft= int(1024)
        self.hop_length= int(self.n_fft/4) #
        self.top_db = 80
        self.fmin = 20
        self.fmax = 8300
        self.sr = int(22050 * 1.0)
       
        
        for path in tqdm(self.wavs_path):
            print(path)
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
            self.data.append(self.spec_to_image(spec_db))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def spec_to_image(self, spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

class Model(nn.Module):
    def __init__(self, input_shape, batch_size=16, num_category=20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.dense1 = nn.Linear(256*(((input_shape[1]//2)//2)//2)*(((input_shape[2]//2)//2)//2),500)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(500, num_category)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2) 
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x 

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
            #print(height, width)
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

    PATH = os.path.join(os.path.abspath('.'), 'parameters.pth')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)


wavs_path = glob(os.path.join('/home/lab/Documents/Human/Elevator_Sound_Classification/Record', '*', '*'))
train_data = Data(wavs_path)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

shape = train_data.__getitem__(0)[0].shape
model = Model(input_shape=(1,shape[0],shape[1]), batch_size=16, num_category=20).to(device)



loss_fn = nn.CrossEntropyLoss()
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 60
train_losses = []

train(model, loss_fn, train_loader, epochs, optimizer, train_losses, lr_decay)

          
        