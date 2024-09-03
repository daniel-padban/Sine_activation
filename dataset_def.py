import numpy as np
import torch
from torch.utils.data import Dataset
import wandb

class SineData(Dataset):
    def __init__(self, noise_std, start, end, data_len):
        self.noise_std = noise_std
        self.start = start
        self.end = end
        self.data_len = data_len
        self.X_tensor, raw_y = self.generate_data()
        self.y_tensor = self.add_noise(raw_y=raw_y)
        
    def generate_data(self):
        x = torch.linspace(self.start, self.end, self.data_len)
        y = np.sin(x)
        return x, y
    
    def _add_noise(self,raw_y:torch.Tensor): #add gaussian noise - regulate with 'noise_std'
        noise = torch.normal(mean = 0, std=self.noise_std, size=raw_y.shape)
        noisy_y = raw_y + noise
        return noisy_y
    
    def create_artifact(self):
        dataset_art = wandb.Artifact(type='dataset')
        metadata_dict = {
            "start":self.start,
            "end":self.end,
            "data_len":self.data_len,
            "noise_std":self.noise_std
        }
        dataset_art.metadata = metadata_dict
        return dataset_art
    
    def __len__(self):
        return self.y_tensor.size(0)

    def __getitem__(self, idx):
        X = self.X_tensor[idx]
        y = self.y_tensor[idx]
        return X, y
    
