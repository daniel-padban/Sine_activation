import wandb
import torch
import torch.optim as optim
import torchmetrics
import json

def json2dict(json_path): 
    with open(json_path,'r') as json_file:
        dict = json.load(json_file)
        return dict

class WandbTrainer():
    def __init__(self,run:wandb,model:torch.nn.Module,test_dataloader, train_dataloader, device) -> None:

        self.run = run
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = self._get_loss()
        self.optim_fn = self._get_optim()
        self.device = device
        self.n_shift = self.run.config['n_shift']
        self.output_len = self.run.config['output_len']


    def _get_loss(self):
        loss_fn_dict ={
            'mse':torch.nn.MSELoss,
            'mape':torchmetrics.MeanAbsolutePercentageError,
        }
        loss_key = self.run.config["loss"].lower()
        if loss_key not in loss_fn_dict:
            raise ValueError(f'config param "loss" is not a valid option. Recieved: "{loss_key}"')
        elif loss_key is None:
            raise ValueError(f'config param "loss" does not exist in config dict')
        
        loss_fn = loss_fn_dict.get(loss_key)
        return loss_fn

    def _get_optim(self):
        optim_dict = {
            'sgd':optim.SGD, # with nesterov
            'adam':optim.Adam, # proposed - lr: 3e-4
            'adamw':optim.AdamW,
            'adadelta':optim.Adadelta,
            'adagrad':optim.Adagrad
        }
        optim_key = self.run.config['optim'].lower()
        if optim_key not in optim_dict:
            raise ValueError(f'config param "optim" is not a valid option. Recieved: "{optim_key}"')
        elif optim_key is None:
            raise ValueError(f'config param "optim" does not exist in config dict')
        
        optim_fn = optim_dict[optim_key]
        return optim_fn
    
    def _train_loop(self):
        lr = self.run.config['lr']
        w_decay = self.run.config['w_decay']
        optimizer = self.optim_fn(params=self.model.parameters(recurse=True),lr=lr,w_decay=w_decay)
        for X, y in self.train_dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            last_n_pred = pred[-self.output_len:]
            last_n_y = y[-self.output_len:]
            
            loss = self.loss_fn(last_n_pred,last_n_y)
            loss.backward()
            #update params
            optimizer.step()
            optimizer.zero_grad()
            self.run.log({"Batch train loss":loss.item()}) # log loss to wandb

    def _test_loop(self):
        for X, y in self.test_dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            last_n_pred = pred[-self.output_len:]
            last_n_y = y[-self.output_len:]
            
            loss = self.loss_fn(last_n_pred,last_n_y)
            self.run.log({"Batch test loss":loss.item()})
    
    def full_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            self._train_loop()
            self._test_loop()
            self.run.log({"Epoch": epoch})
            
    def train_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            self._train_loop()
            self.run.log({"Epoch": epoch})

    def test_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            self._test_loop()
            self.run.log({"Epoch": epoch})

