import torch
from torch.utils.data import Dataset
import wandb

class SineData(Dataset):
    def __init__(self, noise_std:float, start:float, end:float, data_len:int,seq_len:int,n_shift:int):
        self.noise_std = noise_std
        self.start = start
        self.end = end
        self.data_len = data_len
        self.seq_len = seq_len
        self.n_shift = n_shift

        X_tensor, raw_y = self._generate_data()
        noisy_y_tensor = self._add_noise(raw_y=raw_y)
        self.shifted_X, self.shifted_y = self.shift_data(X_tensor,noisy_y_tensor)
        sequenced_X, self.sequenced_y = self._sequencing(self.shifted_X,self.shifted_y)
        if sequenced_X.dim() ==2:
            sequenced_X = sequenced_X.unsqueeze(2)
            self.sequenced_y = self.sequenced_y.unsqueeze(2)
        self.scaled_x = self.scale_x(x=sequenced_X)


    def _generate_data(self):
        x = torch.linspace(self.start, self.end, self.data_len)
        y = torch.sin(x)
        return x, y
    
    def _add_noise(self,raw_y:torch.Tensor): #add gaussian noise - regulate with 'noise_std'
        noise = torch.normal(mean = 0, std=self.noise_std, size=raw_y.shape)
        noisy_y = raw_y + noise
        return noisy_y
    
    def shift_data(self, x_tensor, noisy_y,):
        shifted_y = noisy_y[:-self.n_shift]
        shifted_x = x_tensor[self.n_shift:]
        return shifted_x, shifted_y
    def scale_x(self,x):
        scaled_x = torch.log1p(x)
        return scaled_x
    
    def _sequencing(self,x,y):
        num_seqs = self.data_len // self.seq_len
        required_len = (self.seq_len* num_seqs) + self.n_shift # n=n_shift rows are removed already
        while required_len > self.data_len: # handle cases where self.data_len is less than the required len
            num_seqs -= 1 # reduce number of seqs needed
            required_len = (num_seqs * self.seq_len) + self.n_shift 
        
        n_rows_to_drop = self.data_len - required_len
        req_len_X = x[:-n_rows_to_drop]
        req_len_y = y[:-n_rows_to_drop]

        sequenced_X = torch.reshape(req_len_X,[num_seqs,self.seq_len])
        sequenced_y = torch.reshape(req_len_y,[num_seqs,self.seq_len])

        return sequenced_X, sequenced_y
    
    def create_artifact(self,name):
        dataset_art = wandb.Artifact(name=name,type='dataset')
        metadata_dict = {
            "start":self.start,
            "end":self.end,
            "data_len":self.data_len,
            "noise_std":self.noise_std
        }
        dataset_art.metadata = metadata_dict
        return dataset_art
    
    def __len__(self):
        return self.sequenced_y.size(0)

    def __getitem__(self, idx):
        X = self.scaled_x[idx]
        y = self.sequenced_y[idx]
        return X, y
    
