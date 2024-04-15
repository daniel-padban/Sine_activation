import numpy
import torch
import torch.autograd
import pandas as pd
import torch.nn as nn
from data_loader_2 import train_feats,train_target, val_feats, val_target



device = torch.device("cpu")


class SinActivation(nn.Module):
    #sine activation function
    def forward(self,input):
        return torch.sin(input)

class SineNet(nn.Module):
    #model architechture
    def __init__(self,input_size,hidden_size, output_size, stacked_layers):
        super(SineNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = stacked_layers

        self.lstm = nn.LSTM(input_size,hidden_size,stacked_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)

    #feedforward structure
    def forward(self,x ):
        lstm = self.lstm
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers,batch_size,self.hidden_size,)   
        c0 = torch.zeros(self.num_layers,batch_size,self.hidden_size,)

        lstm_out, (hn,cn) = lstm(x, (h0,c0))

        output = self.fc1(lstm_out[:,-1,:])

        return output

model =  SineNet(4,16,1,1).to(device)
print(model)
print(train__feats.shape)

'''input_tensor = xq_train_tensor
print(f"Input tensor:\n{input_tensor}")
print(input_tensor.shape)'''


lossfunc = nn.MSELoss()
optimizer = torch.optim.Adadelta(model.parameters(recurse=True))

learning_rate = 1e-3
batch_size = 73
epochs = 5

result = model(train__feats)

result = result.detach().numpy()
print(f"Result tensor: {result}")
print(result.shape)

result = pd.DataFrame(result)
result.to_csv('test_result.csv')