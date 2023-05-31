import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings

class GPUTransformNeuralfp(nn.Module):
    
    def __init__(self, ir_dir, noise_dir, sample_rate, n_frames=240, train=True, cpu=False):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = sample_rate
        self.ir_dir = ir_dir
        self.n_frames = n_frames
        self.train = train
        self.cpu = cpu
        self.gpu_transform = Compose([
            # ApplyImpulseResponse(ir_paths=self.ir_dir, p=0.5),
            AddBackgroundNoise(background_paths=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.8),
            ])
        
        self.cpu_transform = Compose([
            ApplyImpulseResponse(ir_paths=self.ir_dir, p=0.5),
            # AddBackgroundNoise(background_paths=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.8),
            ])
        
        self.val_transform = Compose([
            ApplyImpulseResponse(ir_paths=self.ir_dir, p=0.5),
            AddBackgroundNoise(background_paths=noise_dir, min_snr_in_db=0, max_snr_in_db=10,p=0.8),
            ])
        
        self.logmelspec = nn.Sequential(
            MelSpectrogram(sample_rate=22050, win_length=740, hop_length=185, n_fft=740, n_mels=128),
            AmplitudeToDB()
        ) 
        self.spec_aug = nn.Sequential(
            TimeMasking(time_mask_param=80),
            FrequencyMasking(freq_mask_param=64)
)

    def forward(self, x_i, x_j):

        if self.cpu:
            x_j = self.cpu_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            return x_i, x_j.flatten()


        if self.train:
            X_i = self.logmelspec(x_i)
            X_i = self.spec_aug(X_i)
            X_i = F.pad(X_i, (self.n_frames - X_i.size(-1), 0))

            x_j = self.gpu_transform(x_j, sample_rate=self.sample_rate)
            X_j = self.logmelspec(x_j)
            X_j = self.spec_aug(X_j)
            X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0)) 


            return X_i.permute(0,1,3,2), X_j.permute(0,1,3,2)
        
        else:
            print(x_i.shape)
            X_i = self.logmelspec(x_i).permute(2,0,1)
            X_i = X_i.unfold(0, size=self.n_frames, step=self.n_frames//2).permute(1,0,3,2)

            x_j = self.val_transform(x_j, sample_rate=self.sample_rate)
            X_j = self.logmelspec(x_j).permute(2,0,1)
            X_j = X_j.unfold(0, size=self.n_frames, step=self.n_frames//2).permute(1,0,3,2)

            
            return X_i, X_j