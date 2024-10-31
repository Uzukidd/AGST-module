import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from captum.attr import visualization, GradientAttribution, LayerAttribution

from typing import *

from . import attack_utils
from .model_utils import *

class ASGT:
    def __init__(self, 
                 model:nn.Module,
                 training_forward_func:Callable,
                 loss_func:nn.Module,
                 attak_func:Union[Callable, str],
                 explain_func:Callable,
                 eps:float,
                 k:int,
                 lam:float,
                 feature_range:Union[list, tuple],
                 device:torch.device, 
                 preprocess:Optional[transforms.Compose] = None) -> None:
        if isinstance(attak_func, str):
            attak_func = getattr(attack_utils, attak_func)(model = model, 
                                                           loss_func = loss_func,
                                                           eps = eps)
        self.model = model
        
        self.training_forward_func = training_forward_func
        self.loss_func = loss_func
        self.attak_func = attak_func
        self.explain_func = explain_func
        self.preprocess = preprocess
        
        self.k = k
        self.lam = lam
        self.feature_range = feature_range
        self.device = device
        
    def _asgt_iterate(self, 
                      batch_X:torch.Tensor,
                      batch_adv_X:torch.Tensor,
                      masked_batch_adv_X:torch.Tensor,
                      batch_Y:torch.Tensor):
        
        self.model.train()
        clean_logit = self.model(batch_X)
        adv_logit = self.model(batch_adv_X)
        masked_adv_logit = self.model(masked_batch_adv_X)
        
        clean_prob_log = nn.functional.log_softmax(clean_logit, dim=1)
        masked_adv_prob_log = nn.functional.log_softmax(masked_adv_logit, dim=1)
        
        loss = self.loss_func(clean_logit, batch_Y) \
                + self.loss_func(adv_logit, batch_Y) \
                + self.lam * nn.functional.kl_div(clean_prob_log, masked_adv_prob_log, reduction='batchmean', log_target=True)
        
        if self.training_forward_func is not None:
            self.training_forward_func(loss)
        
        return loss.item()
        
    def evaluate_model(self, data_loader):
        return attack_utils.evaluate_model(self.model, 
                                           data_loader,
                                           self.device)
        
    def evaluate_model_robustness(self, data_loader):
        return attack_utils.evaluate_model_robustness(self.model,
                                               data_loader,
                                               self.attak_func,
                                               self.device)
        
    def generate_masked_adv_sample(self, batch_X:torch.Tensor,
                                    batch_Y:torch.Tensor):
        B, C, W, H = batch_X.size()
        self.model.eval()
        batch_adv_X:torch.Tensor = self.attak_func(batch_X, 
                                                    batch_Y)
        
        attributions:torch.Tensor = self.explain_func(batch_adv_X, target = batch_Y)
        attributions = attributions.mean(1).abs().view(B, W * H)
        _, attributions_masked_indices = torch.topk(attributions, self.k, dim=1,largest=False)
        attributions_masked_indices = attributions_masked_indices.unsqueeze(1).expand(-1, C, -1)
        
        _random_values = attributions.new_empty(attributions_masked_indices.size()).uniform_(*self.feature_range)
        masked_batch_adv_X = batch_adv_X.detach().clone().view(B, C, -1)
        masked_batch_adv_X.scatter_(2, attributions_masked_indices, _random_values)
        masked_batch_adv_X = masked_batch_adv_X.view(B, C, W, H)

        return batch_adv_X, masked_batch_adv_X
    
    @staticmethod
    def show_image(images:torch.Tensor, comparison_images:torch.Tensor=None):
        import torchvision
        import matplotlib.pyplot as plt
        
        if comparison_images is not None:
            images = torch.cat((images, comparison_images), dim=3)
        
        images = images.detach().cpu()
        grid_img = torchvision.utils.make_grid(images, nrow=2, normalize=True)

        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        import pdb; pdb.set_trace()
        
    def train_one_epoch(self, data_loader:DataLoader, use_tqdm=True):
        running_loss = 0
        
        data_loader
        if use_tqdm:
            data_loader = tqdm(data_loader, 
                            total=data_loader.__len__())

        for data in data_loader:
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)
            
            batch_adv_X, masked_batch_adv_X = self.generate_masked_adv_sample(batch_X, batch_Y)
            # self.show_image(batch_X, batch_adv_X)
            loss = self._asgt_iterate(batch_X, 
                                batch_adv_X,
                                masked_batch_adv_X, 
                                batch_Y)
            running_loss += loss
            
        return running_loss
    

        