import torch.nn as nn
from custom_act_lstm import CustomLSTMLayer

class SineNet(nn.Module):
    def __init__(self, input_size, hidden_size, activation:object):
        '''
        :output gates: (ft, it, ot)
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = CustomLSTMLayer(input_size=input_size,hidden_size=hidden_size,activation=activation)
        self.fco = nn.Linear(in_features=hidden_size,out_features=1) 

    def forward(self, x):
       
        lstm1_out,(_,ct), gates = self.lstm1(x)
        output = self.fco(lstm1_out)
        return output, ct, gates