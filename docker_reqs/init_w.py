import torch.nn as nn
from torch.nn import init



def custom_init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                init.zeros_(param)