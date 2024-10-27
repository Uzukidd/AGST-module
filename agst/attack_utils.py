import torch
import torch.nn as nn

from autoattack import AutoAttack

from typing import *

class auto_attack(nn.Module):
    def __init__(self, model:nn.Module,
                 eps:float) -> None:
        super().__init__()
        self.adversary = AutoAttack(model, 
                           norm='Linf', 
                           eps=eps, 
                           version='standard',
                           verbose=False)
        self.model = model
    
    def forward(self,
                batch_X:torch.Tensor, 
                batch_Y:torch.Tensor):
        B = batch_X.size(0)
        self.model.zero_grad()
        
        return self.adversary.run_standard_evaluation(batch_X, batch_Y, bs=B)
    
class FGSM(nn.Module):
    def __init__(self, model:nn.Module,
                 eps:float) -> None:
        super().__init__()
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.eps = eps
    
    def forward(self,
                batch_X:torch.Tensor, 
                batch_Y:torch.Tensor):
        B = batch_X.size(0)
        if not batch_X.requires_grad:
            batch_X.requires_grad_(True)
        
        loss = self.loss_func.forward(self.model(batch_X), batch_Y)
        self.model.zero_grad()
        loss.backward()
        
        return (batch_X + self.eps * batch_X.grad.sign()).clamp_(0.0, 1.0).detach()

        