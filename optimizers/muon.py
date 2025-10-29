import torch
import math
from config import *
from optimizers.utils.matrix_decom import zeropower_via_newtonschulz5
import math
from einops import rearrange

class muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (skipped for now)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, 
                params, 
                learning_rate = 1e-2,
                momentum = 0.9,
                weight_decay = 0.1,
                ):
        defaults = dict(lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                assert p.ndim > 1 , 'Muon should not be used for 1D parameters or embeddings; use AdamW instead.'
                grad = p.grad
                # Get state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['step'] = 0
                    if p.ndim == 1:
                        state['v'] = torch.zeros_like(grad)
                
                # Apply momentum
                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - group['momentum'])

                # Skip the Nesterov momentum here to make the comparison fair
                # if group['nesterov']:
                #     update = grad.lerp(buf, group['momentum'])
                # else:
                #     update = buf.clone()
                
                # Apply orthogonalization only for parameters with ndim >= 2
                if buf.ndim == 4:
                    buf = buf.view(len(buf), -1)
                    
                # Apply Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(buf, steps = 5)
                
                # Reshape back if needed
                if p.ndim == 4:
                    update = update.view(p.shape)
                state['step'] += 1
                # Apply RMS norm scaling
                update = (0.2 * math.sqrt(max(p.size(-2), p.size(-1))) * update)           
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply the update
                p.add_(update, alpha=-group['lr'])

        return