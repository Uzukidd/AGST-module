import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
import torchvision.transforms as transforms
import kornia as K
from functools import partial
from torchvision.models import resnet18, alexnet
from torch.utils.data import DataLoader

import os
from enum import Enum

class ComposedModel(nn.Module):
    def __init__(self, model: nn.Module, compose: nn.Module):
        super().__init__()
        self.model = model
        self.compose = compose

    def forward(self, x):
        x = self.compose(x)
        return self.model(x)

def load_model(model_name:str, path:str, device:torch.device, transform:nn.Module=None, num_classes:int=10):
    if model_name == "resnet18":
        model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "alexnet":
        model = alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    model = model.float().to(device)
    
    have_loaded_weights = False
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        have_loaded_weights = True
        print(f"Successfully load weights from \"{path}\"")
        
    if transform is not None:
        model = ComposedModel(model, 
                        transform)
    
    return model, have_loaded_weights