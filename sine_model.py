import torch
import torch.nn as nn
from custom_act_lstm import CustomLSTMLayer

class SineNet(nn.module):
    def __init__(self, input_size, hidden_size, activation:object):
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = CustomLSTMLayer(input_size=input_size,hidden_size=hidden_size,activation=activation)
        self.lstm2 = CustomLSTMLayer(input_size=input_size,hidden_size=hidden_size,activation=activation)
        self.lstm3 = CustomLSTMLayer(input_size=input_size,hidden_size=hidden_size,activation=activation)
        self.fco = nn.Linear(in_features=input_size,out_features=1)

    def forward(self, x):
        lstm1_out,(h1,c1) = self.lstm1(x)
        lstm2_out,(h2,c2) = self.lstm1(lstm1_out,(h1,c1))
        lstm3_out,(_,_) = self.lstm1(lstm2_out,(h2,c2))

        output = self.fco(lstm3_out)
        return output
    

