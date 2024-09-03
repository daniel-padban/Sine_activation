import datetime
import wandb
from trainer_def import WandbTrainer, json2dict
from dataset_def import SineData
from sine_model import SineNet
import torch
from torch.utils.data import DataLoader

config_dict = json2dict('config.json')
datetime_now = datetime.datetime.now()
now_str = datetime.datetime.strftime(datetime_now,"%Y%m%d%H%M%S")
run_id = now_str
run = wandb.init(project='Sine-Gates',config=config_dict)

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
hidden_size= run.config['hidden_size']
model = SineNet(input_size=1,hidden_size=hidden_size,activation=activation)

run.watch(model, log_freq=100) #gradients & model parameters

#data params
noise_std = run.config['noise_std']
train_len = run.config['train_data_len']
train_start = run.config['train_start']
train_end = run.config['train_end']

test_len = run.config['test_data_len']
test_start = run.config['test_start']
test_end = run.config['test_end']

train_dataset = SineData(noise_std=noise_std,
                         start=train_start,
                         end=train_end,
                         data_len=train_len)
test_dataset = SineData(noise_std=noise_std,
                        start=test_start,
                        end=test_end,
                        data_len=test_len)

#dataset artifacts
train_artifact = train_dataset.create_artifact()
test_artifact = test_dataset.create_artifact()
run.log_artifact(train_artifact)
run.log_artifact(test_artifact)

batch_size = run.config['batch_size']
batch_size = run.config['seq_len']

#dataloaders
train_dataloader = DataLoader(train_dataset,batch_size=)


trainer = WandbTrainer(run,model=model,test_dataloader=)