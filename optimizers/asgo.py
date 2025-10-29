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

    
class asgo(torch.optim.Optimizer):
    #  function initializes the optimizer with the model's parameters and some hyperparameters from args.
    def __init__(self, 
                params,
                learning_rate = 1e-2,
                momentum = 0.9, 
                weight_decay = 0.1,
                beta2 = 0.8,
                eps = 1e-10, 
                inv_method = '_matrix_inverse_root_PolarExpress',
                matalg_steps = 10,
                ):
        square_inverse_lib = import_module('optimizers.utils.matrix_decom')
        self._matrix_power = getattr(square_inverse_lib, inv_method)
        self.matalg_steps = matalg_steps
        defaults = {'beta2': beta2, 'lr': learning_rate, 'eps': eps, 
                    'momentum': momentum, 'weight_decay': weight_decay,
                    }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            eps = group['eps']
            momentum = group['momentum']
            beta2 = group['beta2']
            wd = group['weight_decay']
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                grad = p.grad
                assert grad.ndim >= 2, f'1D parameter and embedding should be handled by AdamW'
                if grad.ndim == 2:
                    # Initialize state if needed
                    if len(state) == 0:
                        dim = grad.size(0) if grad.size(0) < grad.size(1) else grad.size(1)
                        state['momentum_buffer'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        state['precond'] = torch.zeros(dim, dim).to(grad.device)
                        state['step'] = 0
                    
                    # Update momentum buffer
                    state['momentum_buffer'].lerp_(grad, 1 - momentum)
                    precond_tmp = torch.mm(grad, grad.t()) if grad.size(0) < grad.size(1) else torch.mm(grad.t(), grad) 
                    state['precond'].lerp_(precond_tmp, 1 - beta2)
                    del precond_tmp
                    
                    # Update inverse square root of preconditioner
                    inverse_precond = self._matrix_power(state['precond'], eps = eps, N = self.matalg_steps)
                    if torch.isnan(inverse_precond).any() or torch.isinf(inverse_precond).any():
                        print(f'[WARNING] Step {state["step"]}: NaN/Inf detected in parameter. Skipping inverse_precond update.')
                        inverse_precond = torch.eye(state['precond'].size(0)).to(state['precond'].device)

                    assert not torch.allclose(inverse_precond, torch.zeros_like(inverse_precond)), f'Inverse precond is all zeros'
                    update = inverse_precond @ state['momentum_buffer'] if grad.size(0) < grad.size(1) else state['momentum_buffer'] @ inverse_precond
                    # Apply preconditioning
                    update.mul_((0.2 * math.sqrt(grad.size(0) * grad.size(1))) / (torch.linalg.matrix_norm(update, ord='fro').item()))
                    state['step'] += 1
                else:
                    raise ValueError('Missing Training Param')
                
                if wd > 0: 
                    p.mul_(1 - group['lr'] * wd)
                p.add_(update, alpha=-group['lr'])
        return
                