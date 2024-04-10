import torch
import pandas as pd
import torch.nn as nn
from multi_preprocessing import\
    xq_train_tensor,y_train_tensor,\
    xq_val_tensor,y_val_tensor,\
    xq_test_tensor,y_test_tensor


class SinActivation(nn.Module):
    def forward(self,input):
        return torch.sin(input)

class SineNet(nn.Module):
    def __init__(self,input_size,hidden_size, stacked_layers, output_size):
        super(SineNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = stacked_layers

        self.lstm = nn.LSTM(input_size,hidden_size,stacked_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        lstm = self.lstm
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers,self.hidden_size)   
        c0 = torch.zeros(self.num_layers,self.hidden_size)

        lstm_out, (hn,cn) = lstm(x, (h0,c0))

        output = self.fc1(lstm_out[:,-1,:])

        return output

model =  SineNet(4,16,1,1)

print(model)

test_run = model(xq_train_tensor)

print(test_run)

test_result = pd.DataFrame(test_run)
test_result.to_csv('test_result.csv')