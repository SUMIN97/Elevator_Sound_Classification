from .library import *



class TestData(Dataset):
    def __init__(self, wavs_path):
       
        self.data = []
        self.wavs_path = wavs_path
        self.n_fft= int(1024)
        self.hop_length= int(self.n_fft/4) #
        self.top_db = 80
        self.fmin = 20
        self.fmax = 8300
        self.sr = int(22050 * 1.0)
               
        for path in self.wavs_path:
            print(path)
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
        return self.data[idx]
    
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