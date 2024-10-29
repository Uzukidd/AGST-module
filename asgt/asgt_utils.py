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
        
    def _asgt_iterrate(self, 
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
        self.model.eval()
        totall_accuracy = []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(data_loader), 
                          total=data_loader.__len__()):
                batch_X, batch_Y = data
                batch_X:torch.Tensor = batch_X.to(self.device)
                batch_Y:torch.Tensor = batch_Y.to(self.device)
                
                outputs = self.model(batch_X)
                predicted = outputs.argmax(1)
                totall_accuracy.append((predicted == batch_Y).float().mean().item())

        totall_accuracy = np.array(totall_accuracy).mean()
        print(f"Accuracy: {100 * totall_accuracy:.2f}%")
        return totall_accuracy
        
    def evaluate_model_robustness(self, data_loader):
        totall_accuracy = []
        for idx, data in tqdm(enumerate(data_loader), 
                          total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)
            
            batch_adv_X:torch.Tensor = self.attak_func(batch_X, 
                                                    batch_Y)
            totall_accuracy.append((self.model(batch_adv_X).argmax(1) == batch_Y).float().mean().item())
        
        totall_accuracy = np.array(totall_accuracy).mean()
        print(f"Robustness accuracy: {100 * totall_accuracy:.2f}%")
        return totall_accuracy
        
    def generate_masked_adv_sample(self, batch_X:torch.Tensor,
                                    batch_Y:torch.Tensor):
        B, C, W, H = batch_X.size()
        self.model.eval()
        batch_adv_X:torch.Tensor = self.attak_func(batch_X, 
                                                    batch_Y)
        
        attributions:torch.Tensor = self.explain_func(batch_adv_X, target = batch_Y)
        attributions_masked_indices = attributions.view(B, -1).argsort(dim = 1, descending=False)[:, :self.k]
        _random_values = attributions.new_empty(attributions_masked_indices.size()).uniform_(*self.feature_range)
        
        masked_batch_adv_X = batch_adv_X.detach().clone().view(B, -1)
        masked_batch_adv_X[:, attributions_masked_indices] = _random_values
        masked_batch_adv_X = masked_batch_adv_X.view(B, C, W, H)
        
        return batch_adv_X, masked_batch_adv_X
    
    def show_image(self, images:torch.Tensor, comparison_images:torch.Tensor=None):
        import torchvision
        import matplotlib.pyplot as plt
        
        if comparison_images is not None:
            images = torch.cat((images, comparison_images), dim=3)
        
        images = images.detach().cpu()

        # 使用 torchvision.utils.make_grid 将 64 张图片排列成 8x8 的网格
        grid_img = torchvision.utils.make_grid(images, nrow=2, normalize=True)

        # 转换为 NumPy 格式以便用 matplotlib 显示
        plt.imshow(grid_img.permute(1, 2, 0))  # 转换为 [H, W, C]
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        import pdb; pdb.set_trace()
        
    def train_one_epoch(self, data_loader:DataLoader):
        running_loss = 0
        for idx, data in tqdm(enumerate(data_loader), 
                          total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)
            
            batch_adv_X, masked_batch_adv_X = self.generate_masked_adv_sample(batch_X, batch_Y)
            # self.show_image(batch_X, batch_adv_X)
            loss = self._asgt_iterrate(batch_X, 
                                batch_adv_X,
                                masked_batch_adv_X, 
                                batch_Y)
            running_loss += loss
            
        return running_loss
    

        