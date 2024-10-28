import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from captum.attr import visualization, GradientAttribution, LayerAttribution

from typing import *

from . import attack_utils

class AGST:
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
                 device:torch.device) -> None:
        if isinstance(attak_func, str):
            attak_func = getattr(attack_utils, attak_func)(model = model, 
                                                           loss_func = loss_func,
                                                           eps = eps)
        self.model = model
        self.training_forward_func = training_forward_func
        self.loss_func = loss_func
        self.attak_func = attak_func
        self.explain_func = explain_func
        self.k = k
        self.lam = lam
        self.feature_range = feature_range
        self.device = device
        
    def _agst_iterrate(self, 
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
        
        loss = self.loss_func(clean_logit, batch_Y) + \
                self.loss_func(adv_logit, batch_Y) + \
                self.lam * nn.functional.kl_div(clean_prob_log, masked_adv_prob_log, reduction='batchmean', log_target=True)
        
        # import pdb; pdb.set_trace()
        if self.training_forward_func is not None:
            self.training_forward_func(loss)
            # print(loss.item())
        
        return loss.item()
        
        # i_accum = i // args.accum_freq
        # step = num_batches_per_epoch * epoch + i_accum

        # if not args.skip_scheduler:
        #     scheduler(step)

        # images, texts = batch
        # images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        # texts = texts.to(device=device, non_blocking=True)

        # data_time_m.update(time.time() - end)
        # optimizer.zero_grad()

        # if args.accum_freq == 1:
        #     with autocast():
        #         model_out = model(images, texts)
        #         logit_scale = model_out["logit_scale"]
        #         if args.distill:
        #             with torch.no_grad():
        #                 dist_model_out = dist_model(images, texts)
        #             model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
        #         losses = loss(**model_out, output_dict=True)

        #         total_loss = sum(losses.values())
        #         losses["loss"] = total_loss

        #     backward(total_loss, scaler)

        #     # Now, ready to take gradients for the last accum_freq batches.
        #     # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        #     # Call backwards each time, but only step optimizer at the end.
        #     optimizer.zero_grad()
        #     for j in range(args.accum_freq):
        #         images = accum_images[j]
        #         texts = accum_texts[j]
        #         with autocast():
        #             model_out = model(images, texts)

        #             inputs_no_accum = {}
        #             inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
        #             if "logit_bias" in model_out:
        #                 inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

        #             inputs = {}
        #             for key, val in accum_features.items():
        #                 accumulated = accum_features[key]
        #                 inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

        #             losses = loss(**inputs, **inputs_no_accum, output_dict=True)
        #             del inputs
        #             del inputs_no_accum
        #             total_loss = sum(losses.values())
        #             losses["loss"] = total_loss

        #         backward(total_loss, scaler)

        # if scaler is not None:
        #     if args.horovod:
        #         optimizer.synchronize()
        #         scaler.unscale_(optimizer)
        #         if args.grad_clip_norm is not None:
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #         with optimizer.skip_synchronize():
        #             scaler.step(optimizer)
        #     else:
        #         if args.grad_clip_norm is not None:
        #             scaler.unscale_(optimizer)
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #         scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     if args.grad_clip_norm is not None:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #     optimizer.step()

        # # reset gradient accum, if enabled
        # if args.accum_freq > 1:
        #     accum_images, accum_texts, accum_features = [], [], {}

        # # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        # with torch.no_grad():
        #     unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # batch_time_m.update(time.time() - end)
        # end = time.time()
        # batch_count = i_accum + 1
        # if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
        #     batch_size = len(images)
        #     num_samples = batch_count * batch_size * args.accum_freq * args.world_size
        #     samples_per_epoch = dataloader.num_samples
        #     percent_complete = 100.0 * batch_count / num_batches_per_epoch

        #     # NOTE loss is coarsely sampled, just master node and per log update
        #     for key, val in losses.items():
        #         if key not in losses_m:
        #             losses_m[key] = AverageMeter()
        #         losses_m[key].update(val.item(), batch_size)

        #     logit_scale_scalar = logit_scale.item()
        #     loss_log = " ".join(
        #         [
        #             f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
        #             for loss_name, loss_m in losses_m.items()
        #         ]
        #     )
        #     samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
        #     samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
        #     logging.info(
        #         f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
        #         f"Data (t): {data_time_m.avg:.3f} "
        #         f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
        #         f"LR: {optimizer.param_groups[0]['lr']:5f} "
        #         f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
        #     )
            
        #     # resetting batch / data time meters per log window
        #     batch_time_m.reset()
        #     data_time_m.reset()
        
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
        
    def train_one_epoch(self, data_loader:DataLoader):
        running_loss = 0
        for idx, data in tqdm(enumerate(data_loader), 
                          total=data_loader.__len__()):
            batch_X, batch_Y = data
            batch_X:torch.Tensor = batch_X.to(self.device)
            batch_Y:torch.Tensor = batch_Y.to(self.device)
            
            batch_adv_X, masked_batch_adv_X = self.generate_masked_adv_sample(batch_X, batch_Y)
             
            loss = self._agst_iterrate(batch_X, 
                                batch_adv_X,
                                masked_batch_adv_X, 
                                batch_Y)
            running_loss += loss
            
        return running_loss
    

        