import numpy
import torch
import pandas as pd
import torch.nn as nn
from multi_preprocessing import\
    xq_train_tensor,y_train_tensor,\
    xq_val_tensor,y_val_tensor,\
    xq_test_tensor,y_test_tensor


device = torch.device("cpu")


class SinActivation(nn.Module):
    #sine activation function
    def forward(self,input):
        return torch.sin(input)

class SineNet(nn.Module):
    #model architechture
    def __init__(self,input_size,hidden_size, stacked_layers, output_size):
        super(SineNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = stacked_layers

        self.lstm = nn.LSTM(input_size,hidden_size,stacked_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)

    #feedforward structure
    def forward(self, x):
        lstm = self.lstm
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers,self.hidden_size)   
        c0 = torch.zeros(self.num_layers,self.hidden_size)

        lstm_out, (hn,cn) = lstm(x, (h0,c0))

        output = self.fc1(lstm_out[:,:])

        return output

model =  SineNet(4,16,1,1).to(device)
print(model)
input_tensor = xq_train_tensor

print(f"Input tensor:\n{input_tensor}")
print(input_tensor.shape)


lossfunc = nn.MSELoss()
#lossfunc.backward()

learning_rate = 1e-3
batch_size = 73
epochs = 5



result_tensor = model(xq_train_tensor)
#result = result_tensor.detach().numpy()
print(f"Result tensor: {result_tensor}")
print(result_tensor.shape)

#test_result = pd.DataFrame(result)
#test_result.to_csv('test_result.csv')