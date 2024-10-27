import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from captum.attr import visualization, GradientAttribution, LayerAttribution

from typing import *

from . import attack_utils

class AGST:
    def __init__(self, 
                 model:nn.Module,
                 attak_func:Union[Callable, str],
                 explain_func:Callable,
                 eps:float,
                 device:torch.device) -> None:
        if isinstance(attak_func, str):
            attak_func = getattr(attack_utils, attak_func)(model = model, 
                                                           eps = eps)
        self.model = model
        self.attak_func = attak_func
        self.explain_func = explain_func
        self.device = device
        
    def train_one_epoch(self, 
                        data_loader:DataLoader):
        totall_accuracy = []
        for idx, data in tqdm(enumerate(data_loader), 
                          total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)

            batch_adv_X:torch.Tensor = self.attak_func(batch_X, 
                                                       batch_Y)
            attributions:torch.Tensor = self.explain_func(batch_adv_X, target = batch_Y)
            
            totall_accuracy.append((self.model(batch_adv_X).argmax(1) == batch_Y).float().mean().item())
        totall_accuracy = np.array(totall_accuracy).mean()
        print(f"totall_accuracy:{totall_accuracy}")