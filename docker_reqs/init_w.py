import torch.nn as nn
from torch.nn import init
from custom_act_lstm import CustomLSTMLayer


def custom_init_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize Linear layer weights
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, CustomLSTMLayer):
        for name, param in m.named_parameters():
            if 'w_ih' in name or 'w_hh' in name:
                init.xavier_uniform_(param)
            elif 'b_hh' in name or 'b_ih' in name:
                init.constant_(param,0)