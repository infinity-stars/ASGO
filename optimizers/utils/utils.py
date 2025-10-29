from importlib import import_module
from optimizers.utils.scheduler import *
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
import time
import logging
from torch import distributed as dist
import os
import glob
from optimizers.asgo import asgo
from optimizers.dasgo import dasgo
from optimizers.shampoo import shampoo
from optimizers.muon import muon

class MultiOptimizer:
    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
    
    def step(self, closure = None):
        for opt in self.optimizers:
            opt.step(closure)
    
    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none)
    
    def state_dict(self):
        state_dict = {}
        for i,opt in enumerate(self.optimizers):
            state_dict[f'optimizer_{i}'] = opt.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            if f'optimizer_{i}' in state_dict:
                opt.load_state_dict(state_dict[f'optimizer_{i}'])
    
    def add_param_group(self, param_group):
        if self.optimizers:
            return self.optimizers[0].add_param_group(param_group)
    
    def __getattr_(self, name):
        for opt in self.optimizers:
            if hasattr(opt, name):
                return getattr(opt, name)
        raise AttributeError(f"Optimizer {self.optimizers[0].__class__.__name__} has no attribute {name}")

def get_lr_scheduler(optimizer, cfg, total_steps):

    # Warmup from near-zero (1/warmup_steps of target lr) to full learning rate
    warmup_steps = total_steps // 10
    warmup_start_factor = 1.0 / warmup_steps
    warmupscheduler = scheduler.LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps)
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max = (total_steps - warmup_steps), eta_min = 0.02 * cfg.optimizer.learning_rate)

    lr_scheduler = scheduler.SequentialLR(
        optimizer,
        schedulers = [warmupscheduler, lr_scheduler],
        milestones = [warmup_steps]
    )
    return lr_scheduler

def get_optimizer_and_lr_scheduler(model, cfg, total_steps):
    Adam_params = []
    other_params = []
    other_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'emb' in name.lower() or param.ndim == 1:
                Adam_params.append(param)
            else:
                other_params.append(param)
                other_names.append(name)
    assert len(Adam_params) >= 1, "No Adam parameters found in the model."
    optimizers_list = []
    if Adam_params:
        Adam_optimizer = torch.optim.Adam(
            Adam_params,
            lr = cfg.optimizer.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            eps=1e-8,
        )
        optimizers_list.append(Adam_optimizer)

    if other_params:
        if cfg.optimizer.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                other_params,
                lr = cfg.optimizer.learning_rate,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                weight_decay=cfg.optimizer.weight_decay,
                eps=cfg.optimizer.damping,
            )
        elif cfg.optimizer.optimizer_name == 'asgo':
            optimizer = asgo(
                other_params,
                learning_rate = cfg.optimizer.learning_rate,
            )
        elif cfg.optimizer.optimizer_name == 'dasgo':
            optimizer = dasgo(
                other_params,
                learning_rate = cfg.optimizer.learning_rate,
            )
        elif cfg.optimizer.optimizer_name == 'shampoo':
            optimizer = shampoo(
                other_params,
                learning_rate = cfg.optimizer.learning_rate,
            )
        elif cfg.optimizer.optimizer_name == 'muon':
            optimizer = muon(
                other_params,
                learning_rate = cfg.optimizer.learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer.optimizer_name}")
        optimizers_list.append(optimizer)
    optimizer = MultiOptimizer(optimizers_list)
    lr_scheduler_list = []
    for opt in optimizer.optimizers:
        lr_scheduler = get_lr_scheduler(opt, cfg, total_steps = total_steps)
        lr_scheduler_list.append(lr_scheduler)
    return optimizer, lr_scheduler_list