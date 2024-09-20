import torch

import torch.nn as nn
import torch.optim as optim
from ka_SineNet import SineNet
import wandb
from trainer_def import json2dict
from init_w import custom_init_weights
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
print(device)

config_dict = json2dict('config.json')
name_str = f"Run-{config_dict['activation']}-S{args.seed}"
run = wandb.init(project='Sine-Gates',config=config_dict,group='Ka-L5-tanh', name=name_str, notes='L5 - hidden size = 10 -> 30, sin + 3rd harmonic + 5th harmonix')

run.config['seed'] = args.seed

step_size = run.config['step_size']

train_start = run.config['train_start']
train_end = run.config['train_end']
X_train = torch.arange(train_start,train_end,step_size)
y_train = np.sin(X_train) + np.sin(3*X_train)/3 + np.sin(5*X_train)/5

n_features = 1

train_series = y_train

#train data
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

def train_loop_sub(epoch,model:nn.Module,optimizer,run,train_dataset,train_labels):
    model.train(True)
    loss_total = 0
    
    model.zero_grad()
    optimizer.zero_grad()
    predictions, ct, gates = model(train_dataset)

    (ft, it, ot) = gates
    ft_mean = ft.mean().item()
    it_mean = it.mean().item()
    ot_mean = ot.mean().item()

    ft_hist = wandb.Histogram(ft.tolist())
    it_hist = wandb.Histogram(it.tolist())
    ot_hist = wandb.Histogram(ot.tolist())

    loss = loss_function(predictions, train_labels)
    loss_total += loss.item()
    loss.backward()
    optimizer.step()

    run.log({"epoch": epoch,
            "train_loss":loss.item(),
            "train_ct":ct.mean().item(),
           
            "train_ft":ft_hist,
            "train_ft_mean":ft_mean,
           
            "train_it":it_hist,
            "train_it_mean":it_mean,
            
            "train_ot":ot_hist,
            "train_ot_mean":ot_mean,

            })


test_end = run.config['test_end']
test_start = run.config['test_start']
X_test = torch.arange(test_start,test_end,step_size)
y_test = np.sin(X_test) + np.sin(3*X_test)/3 + np.sin(5*X_test)/5

test_series = y_test

test_dataset = []
test_labels = []
for i in range(len(test_series)-look_back):
    test_dataset.append(test_series[i:i+look_back])
    test_labels.append(test_series[i+look_back])
test_dataset = torch.stack(test_dataset).unsqueeze(0).to(device=device)
test_labels = torch.stack(test_labels).unsqueeze(0).unsqueeze(2).to(device=device)

def test_loop_sub(epoch,model:nn.Module,run,test_dataset,test_labels):
        model.eval()
        loss_total = 0

        with torch.no_grad():    
            predictions, ct, gates = model(test_dataset)
            
            loss = loss_function(predictions, test_labels)
            loss_total += loss.item()

            (ft, it, ot) = gates
            ft_mean = ft.mean().item()
            it_mean = it.mean().item()
            ot_mean = ot.mean().item()

            ft_hist = wandb.Histogram(ft.tolist())
            it_hist = wandb.Histogram(it.tolist())
            ot_hist = wandb.Histogram(ot.tolist())
            run.log({"epoch": epoch,
                    "test_loss":loss.item(),
                    "test_ct":ct.mean().item(),
           
                    "test_ft":ft_hist,
                    "test_ft_mean":ft_mean,
           
                    "test_it":it_hist,
                    "test_it_mean":it_mean,
           
                    "test_ot":ot_hist,
                    "test_ot_mean":ot_mean,})

for epoch in range(run.config['n_epochs']):
    print(f'---------- Epoch: {epoch+1} ----------')
    train_loop_sub(epoch=epoch,model=model,optimizer=optimizer,run=run,train_dataset=train_dataset,train_labels=train_labels)
    test_loop_sub(epoch=epoch,model=model,run=run,test_dataset=test_dataset,test_labels=test_labels)

run.finish(exit_code=0)