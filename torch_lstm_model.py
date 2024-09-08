
import torch.nn as nn

class TorchLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True)
        self.fco = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, x):
        lstm1_out,(_,_) = self.lstm1(x)
        output = self.fco(lstm1_out)
        return output
    

