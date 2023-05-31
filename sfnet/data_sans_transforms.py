import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import warnings
import torch.nn as nn
import warnings


from util import load_index, get_frames, qtile_normalize

clip_len = 2.0
SAMPLE_RATE = 22050

class NeuralfpDataset(Dataset):
    def __init__(self, path, n_frames=240, offset=0.2, norm=0.95, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = norm
        self.offset = offset 
        self.n_frames = n_frames
        self.filenames = load_index(path)

        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.filenames[str(idx)]
        try:
            audio, sr = torchaudio.load(datapath)

        except Exception:

            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        if self.norm is not None:
            audio_mono = qtile_normalize(audio_mono, q=self.norm)
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio_resampled = resampler(audio_mono)    

        clip_frames = int(SAMPLE_RATE*clip_len)
        
        if len(audio_resampled) <= clip_frames:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        
        #   For training pipeline, output a random frame of the audio
        if self.train:
            offset_mod = int(SAMPLE_RATE*(self.offset) + clip_frames)
            if len(audio_resampled) < offset_mod:
                print(len(audio_resampled), offset_mod)
            r = np.random.randint(0,len(audio_resampled)-offset_mod)
            ri = np.random.randint(0,offset_mod - clip_frames)
            rj = np.random.randint(0,offset_mod - clip_frames)
            clip = audio_resampled[r:r+offset_mod]
            x_i = clip[ri:ri+clip_frames]
            x_j = clip[rj:rj+clip_frames]

            if self.transform is not None:
                x_i, x_j = self.transform(x_i, x_j)

            return torch.unsqueeze(x_i, 0), torch.unsqueeze(x_j, 0)
        
        #   For validation / test, output consecutive (overlapping) frames
        else:
            return torch.unsqueeze(audio_resampled, 0)
            # return audio_resampled
    
    def __len__(self):
        return len(self.filenames)