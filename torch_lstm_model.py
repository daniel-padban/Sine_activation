
import torch.nn as nn

class TorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,batch_first=True)
        self.fco = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, x):
        lstm1_out,(h1,c1) = self.lstm1(x)
        lstm2_out,(_,_) = self.lstm2(lstm1_out,(h1,c1))
        output = self.fco(lstm2_out)
        return output
    

