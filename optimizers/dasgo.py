import torch
from optimizers.utils import *
from config import *
from einops import rearrange
from optimizers.utils.utils import *
from optimizers.utils.matrix_decom import zeropower_via_newtonschulz5
import pdb
import math
from importlib import import_module
import logging

    
class dasgo(torch.optim.Optimizer):
    #  function initializes the optimizer with the model's parameters and some hyperparameters from args.
    def __init__(self, 
                params, 
                learning_rate = 1e-3,
                momentum = 0.9,
                weight_decay = 0.1,
                beta2 = 0.95,
                eps = 1e-8,
                ):
        defaults = {'beta2': beta2, 'lr': learning_rate, 'eps': eps,
                    'momentum': momentum, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            eps = group['eps']
            momentum = group['momentum']
            lr = group['lr']
            beta2 = group['beta2']
            wd = group['weight_decay']
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                grad = p.grad.data
                assert grad.ndim >= 2, f'1D parameter and embedding should be handled by AdamW'
                if len(state) == 0:
                    dim = grad.size(1)
                    state['momentum_buffer'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                    state['precond'] = torch.zeros(dim).to(grad.device)
                    state['step'] = 0
                state['momentum_buffer'].lerp_(grad, 1 - momentum)
                state['precond'].lerp_(torch.sum(grad * grad, dim = 0, keepdim = False), 1 - beta2)
                state['step'] += 1
                update = state['momentum_buffer'] * state['precond'].add(eps).pow(-0.5)
                # Do the RMS norm scaling
                update.mul_((0.2 * math.sqrt(grad.size(0) * grad.size(1))) / (torch.linalg.matrix_norm(update, ord='fro').item()))
                if wd > 0:
                    p.data.mul_(1 - wd * lr)
                p.data.add_(update, alpha=-group['lr'])
        return