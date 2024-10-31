import torch
import torch.nn as nn
import numpy as np

from autoattack import AutoAttack

from typing import *

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from tqdm import tqdm

class auto_attack(nn.Module):
    def __init__(self, model:nn.Module,
                loss_func:nn.Module,
                 eps:float) -> None:
        super().__init__()
        self.adversary = AutoAttack(model, 
                           norm='Linf', 
                           eps=eps, 
                           version='standard',
                           verbose=False)
        self.model = model
        self.loss_func = loss_func
        self.eps = eps
    
    def forward(self,
                batch_X:torch.Tensor, 
                batch_Y:torch.Tensor):
        B = batch_X.size(0)
        batch_X = batch_X.detach()
        self.model.zero_grad()
        
        return self.adversary.run_standard_evaluation(batch_X, batch_Y, bs=B)
    
class FGSM(nn.Module):
    def __init__(self, model:nn.Module,
                 loss_func:nn.Module,
                 eps:float) -> None:
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.eps = eps
    
    def forward(self,
                batch_X:torch.Tensor, 
                batch_Y:torch.Tensor):
        B = batch_X.size(0)
        batch_X = batch_X.clone().detach().requires_grad_(True)
        
        loss = self.loss_func(self.model(batch_X), batch_Y)
        self.model.zero_grad()
        loss.backward()
        
        return (batch_X + self.eps * batch_X.grad.sign()).clamp_(0.0, 1.0).detach()

class PGD(nn.Module):
    def __init__(self, model:nn.Module,
                 loss_func:nn.Module,
                 eps:float,
                 alpha:float,
                 random_start:bool) -> None:
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.eps = eps
        self.alpha = alpha
        self.random_start = random_start
        
    
    def forward(self,
                batch_X:torch.Tensor, 
                batch_Y:torch.Tensor):
        B = batch_X.size(0)
        batch_X = batch_X.detach().requires_grad_(True)
        
        loss = self.loss_func(self.model(batch_X), batch_Y)
        self.model.zero_grad()
        loss.backward()
        
        return (batch_X + self.alpha * batch_X.grad.sign()).clamp_(batch_X - self.eps, batch_X + self.eps).clamp_(0.0, 1.0).detach()

def evaluate_model(model:nn.Module, 
                   data_loader:torch.utils.data.dataloader, 
                   device:torch.device):
    model.eval()
    totall_accuracy = []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(data_loader), 
                        total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(device)
            batch_Y:torch.Tensor = batch_Y.to(device)
            
            outputs = model(batch_X)
            predicted = outputs.argmax(1)
            totall_accuracy.append((predicted == batch_Y).float().mean().item())

    totall_accuracy = np.array(totall_accuracy).mean()
    print(f"Accuracy: {100 * totall_accuracy:.2f}%")
    return totall_accuracy
        
def evaluate_model_robustness(model:nn.Module, 
                              data_loader:torch.utils.data.dataloader, 
                              attak_func:Callable,
                              device:torch.device):
    model.eval()
    totall_accuracy = []
    batch_adv_X_list = []
    for idx, data in tqdm(enumerate(data_loader), 
                        total=data_loader.__len__()):
        batch_X, batch_Y = data
        batch_X:torch.Tensor = batch_X.to(device)
        batch_Y:torch.Tensor = batch_Y.to(device)
        
        batch_adv_X:torch.Tensor = attak_func(batch_X, 
                                                batch_Y)
        # batch_adv_X_list.append(batch_adv_X.detach().cpu())
        totall_accuracy.append((model(batch_adv_X).argmax(1) == batch_Y).float().mean().item())
    
    # batch_adv_X_list = torch.concat(batch_adv_X_list, 0)
    totall_accuracy = np.array(totall_accuracy).mean()
    print(f"Robustness accuracy: {100 * totall_accuracy:.2f}%")
    return totall_accuracy