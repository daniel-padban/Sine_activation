import torch
from torch.utils.data import Dataset
import wandb

class SineData(Dataset):
    def __init__(self, start:float, end:float, step_size:float,seq_len:int,noise_std = None,):
        self.noise_std = noise_std
        self.start = start
        self.end = end
        self.step_size = step_size
        self.seq_len = seq_len

        X_tensor, raw_y = self._generate_data()
        if noise_std is not None: #should noise be added and how much?
            noisy_y_tensor = self._add_noise(raw_y=raw_y)
            self.sequenced_X, self.sequenced_y = self._sequencing(X_tensor,noisy_y_tensor)
        else:
            self.sequenced_X, self.sequenced_y = self._sequencing(X_tensor,raw_y)

        if self.sequenced_X.dim() ==2:
            self.sequenced_X = self.sequenced_X.unsqueeze(2)
            self.sequenced_y = self.sequenced_y.unsqueeze(2)


    def _generate_data(self):
        x = torch.arange(start=self.start, end=self.end, step=self.step_size)
        y = torch.sin(x)
        return x, y
    
    def _add_noise(self,raw_y:torch.Tensor): #add gaussian noise - regulate with 'noise_std'
        noise = torch.normal(mean = 0, std=self.noise_std, size=raw_y.shape)
        noisy_y = raw_y + noise
        return noisy_y
    
    ''' def scale_x(self,x:torch.Tensor):
        scaler = sklearn.preprocessing.MinMaxScaler((0,1))
        np_x = x.numpy(force=True)
        np_x = np_x.reshape(-1,1) #fit_transform requires reshaping of 1d arrays
        scaled_x = scaler.fit_transform(np_x)
        scaled_x_tensor = torch.tensor(scaled_x)
        self.scaler = scaler
        return scaled_x_tensor'''
    
    def _sequencing(self,x:torch.Tensor,y:torch.Tensor):
        data_len = y.size(0)
        num_seqs = data_len // self.seq_len
        required_len = (self.seq_len* num_seqs)  # n=n_shift rows are removed already
        while required_len > data_len: # handle cases where data_len is less than the required len
            num_seqs -= 1 # reduce number of seqs needed
            required_len = (num_seqs * self.seq_len) + self.n_shift 
        
        n_rows_to_drop = data_len - required_len
        if n_rows_to_drop == 0:
            req_len_X = x
            req_len_y = y
        else:
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
            "step_size":self.step_size,
            "noise_std":self.noise_std
        }
        dataset_art.metadata = metadata_dict
        return dataset_art
    
    def __len__(self):
        return self.sequenced_y.size(0)

    def __getitem__(self, idx):
        X = self.sequenced_X[idx]
        y = self.sequenced_y[idx]
        return X, y