import torch
import math
from config import *
from optimizers.utils.matrix_decom import _matrix_inverse_root_newton
import math
from einops import rearrange

def _matrix_inverse_root_newton(
    A,
    epsilon=1e-10,
    N=50,
    tolerance = 1e-6,
    root=4,
    ):
    # tolerance  tolerance = 0.03
    """Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    NOTE: Exponent multiplier not compatible with coupled inverse Newton iteration!

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 1000)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A_ridge = A.add(identity, alpha=epsilon)

    # initialize matrices
    A_nrm = A_ridge.norm()
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A_ridge
    error = (M - identity).norm() / M.norm()

    # main for loop
    while error > tolerance and iteration < N:
        iteration += 1
        M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = (M - identity).norm() / M.norm()

    if torch.isnan(X).any() or torch.isinf(X).any():
        print(f'[WARNING] NaN/Inf detected in _matrix_inverse_root_newton. Returning identity matrix.')
        X = torch.eye(len(X)).to(X.device)

    return X


class shampoo(torch.optim.Optimizer):
    r"""Implements Shampoo Optimizer Algorithm.

    It has been proposed in `Shampoo: Preconditioned Stochastic Tensor
    Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
        epsilon: epsilon added to each mat_gbar_j for numerical stability
            (default: 1e-4)
        update_freq: update frequency to compute inverse (default: 1)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Shampoo(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://arxiv.org/abs/1802.09568

    Note:
        Reference code: https://github.com/moskomule/shampoo.pytorch
    """

    def __init__(self, 
                params,
                learning_rate = 5e-3,
                momentum = 0.9,
                weight_decay = 0.1,
                eps = 1e-10,
                beta2 = 0.999,
                update_freq = 1, 
                inverse_order = 4,
                ):

        defaults = dict(
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=eps,
            update_freq=update_freq,
            beta2=beta2,
            inverse_order=inverse_order
        )
        super().__init__(params, defaults)
            

    def step(self, closure = None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                beta2 = group['beta2']
                eps = group["epsilon"]
                inverse_order = group["inverse_order"]
                if len(state) == 0:
                    state["step"] = 0
                    state['momentum_buffer'] = grad.new_zeros(grad.size())
                    state['v_sq'] = grad.new_zeros(grad.size())
                    for dim_id, dim in enumerate(original_size):
                        state[f'precond_{dim_id}'] = grad.new_zeros(dim, dim)
                        state[f'inv_precond_{dim_id}'] = grad.new_zeros(dim, dim)
                if momentum < 1:
                    state['momentum_buffer'].lerp_(grad, 1 - momentum)
                    state['v_sq'].lerp_(grad * grad.conj(), 1 - beta2)

                # update = state['momentum_buffer'] / (1 - momentum ** (state['step'] + 1))
                update = state['momentum_buffer']
                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(original_size):
                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    update = update.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    # grad = Mat_{dim_id}(grad)
                    state[f"precond_{dim_id}"].lerp_(grad.view(dim, -1) @ grad.view(dim,-1).t(), 1 - beta2)
                    precond = state[f"precond_{dim_id}"]
                    if state["step"] % group["update_freq"] == 0:
                        state[f'inv_precond_{dim_id}'].copy_(_matrix_inverse_root_newton(precond, epsilon = eps, root = inverse_order))
                    inv_precond = state["inv_precond_{}".format(dim_id)]
                    if dim_id == order - 1:
                        # finally
                        update = update.view(dim, -1).t() @ inv_precond
                        # grad: (-1, last_dim)
                        update = update.view(original_size)
                    else:
                        # if not final
                        update = inv_precond @ update.view(dim, -1)
                        # grad (dim, -1)
                        update = update.view(transposed_size)
                state["step"] += 1
                adam_update = state['momentum_buffer'] / state['v_sq'].add(eps).sqrt()
                update = (adam_update.norm() / update.norm()) * update
                if weight_decay > 0:
                    p.data.mul_(1 - weight_decay * group["lr"])
                p.data.add_(update.view(p.data.size()), alpha= -group["lr"])
                

        return