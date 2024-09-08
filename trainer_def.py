import sklearn
import sklearn.preprocessing
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
        optimizer = self.optim_fn(params=self.model.parameters(recurse=True),lr=lr,weight_decay=w_decay)
        loss_ = self.loss_fn() # init loss func
        running_loss = 0
        for X, y in self.train_dataloader:
            optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            last_n_pred = pred[-self.output_len:]
            
            last_n_y = y[-self.output_len:]

            loss = loss_(last_n_pred,last_n_y)
            loss.backward()
            running_loss += loss.item()
            #update params
            optimizer.step()
        mean_train_loss = running_loss / len(self.train_dataloader)
        return mean_train_loss, pred
            

        
    def _test_loop(self):
        loss_ = self.loss_fn() # init loss func
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                last_n_pred = pred[-self.output_len:]
                
                last_n_y = y[-self.output_len:]

                loss = loss_(last_n_pred,last_n_y)
                running_loss += loss.item()
            mean_test_loss = running_loss/len(self.test_dataloader)
            return mean_test_loss, pred
    
    def full_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            train_loss, train_pred = self._train_loop()
            test_loss,test_pred = self._test_loop()
            self.run.log({"Epoch": epoch,"test_loss":test_loss,"train_loss":train_loss})
            print(f"---------- Full epoch: {epoch+1} ----------")
        self.run
            
    def train_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            train_loss,train_pred = self._train_loop()
            self.run.log({"Epoch": epoch,"train_loss":train_loss})
            print(f"---------- Train epoch: {epoch+1} ----------")

    def test_epoch_loop(self):
        epochs = self.run.config['n_epochs']
        for epoch in range(epochs):
            test_loss,test_pred = self._test_loop()
            self.run.log({"Epoch": epoch,"test_loss":test_loss})
            print(f"---------- Test epoch: {epoch+1} ----------")

