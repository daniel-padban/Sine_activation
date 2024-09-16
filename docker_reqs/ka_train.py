import torch
torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
import torch.nn as nn
import torch.optim as optim
from ka_SineNet import SineNet
import wandb
from trainer_def import json2dict
from init_w import custom_init_weights


device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
print(device)

config_dict = json2dict('docker_reqs/config.json')
run = wandb.init(project='Sine-Gates',config=config_dict,group='Ka-L-sin')

step_size = run.config['step_size']

train_start = run.config['train_start']
train_end = run.config['train_end']
X_train = torch.arange(train_start,train_end,step_size)
y_train = torch.sin(X_train)


test_end = run.config['test_end']
test_start = run.config['test_start']
X_test = torch.arange(test_start,test_end,step_size)
y_test = torch.sin(X_test)

n_features = 1

train_series = y_train
test_series  = y_test

look_back = run.config['seq_len']

train_dataset = []
train_labels = []
for i in range(len(train_series)-look_back):
    train_dataset.append(train_series[i:i+look_back])
    train_labels.append(train_series[i+look_back])
train_dataset = torch.stack(train_dataset).unsqueeze(0).to(device=device)
train_labels = torch.stack(train_labels).unsqueeze(0).unsqueeze(2).to(device=device)

n_neurons = run.config['hidden_size']


activation_dict = {
    'tanh':torch.tanh,
    'sin':torch.sin,
    'cos':torch.cos,
    'relu':torch.relu,
    'silu':torch.nn.SiLU(),}
activation_key = run.config['activation'].lower()
if activation_key not in activation_dict.keys():
    raise ValueError(f'Key "activation" in config is not an available option. key: {activation_key}')

activation = activation_dict[activation_key]

model = SineNet(input_size=look_back, hidden_size=n_neurons,activation=activation)

model.apply(custom_init_weights)
model.to(device=device)
model_graph = run.watch(model, log_freq=50,log_graph=True,log='all')

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=run.config['lr'])

loss_curve = []

for epoch in range(run.config['n_epochs']):
    print(f'---------- Epoch: {epoch+1} ----------')
    loss_total = 0
    
    model.zero_grad()
    optimizer.zero_grad()
    predictions = model(train_dataset)
    
    loss = loss_function(predictions, train_labels)
    loss_total += loss.item()
    loss.backward()
    optimizer.step()

    run.log({"Train_epoch": epoch,"train_loss":loss.item()})

run.finish(exit_code=0)