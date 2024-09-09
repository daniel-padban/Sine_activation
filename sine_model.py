import torch
import torch.nn as nn
from custom_act_lstm import CustomLSTMLayer

class SineNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_shift, activation:object):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(in_features=input_size,out_features=hidden_size)
        self.lstm1 = CustomLSTMLayer(input_size=hidden_size,hidden_size=hidden_size,activation=activation)
        self.fco = nn.Linear(in_features=hidden_size,out_features=n_shift) #output = n_shift timesteps

    def forward(self, x):
       
        fc1_out = self.fc1(x)
        lstm1_out,(_,_) = self.lstm1(fc1_out)
        output = self.fco(lstm1_out)
        return output