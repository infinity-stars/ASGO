import torch
from itertools import repeat  

def _matrix_inverse_root_newton(
    A,
    epsilon=1e-6,
    N=100,
    tolerance = 1e-6,
    root=2,
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
    
    if torch.allclose(X, float('inf') * torch.eye(len(X)).to(X.device)):
        # If X is close to zero, return identity matrix
        X = float('nan') * torch.eye(len(X)).to(X.device)

    return X

def _matrix_inverse_root_newtonschulz(G, eps, N = 100, param = (2,-1.5,0.5)):
    # (3.4445, -4.7750, 2.0315)
    assert G.ndim > 1
    dim = G.size(-1)
    a, b, c = param
    I = torch.eye(dim).to(G.device) if G.ndim == 2 else torch.eye(G.size(-1)).expand(len(G),dim,dim).to(G.device)
    # normalize = torch.linalg.matrix_norm(G, ord=2)
    normalize = G.norm() if G.ndim == 2 else G.norm(dim = (-2,-1)).view(len(G),1,1)
    Y = (G + eps * I) / (normalize + eps)
    Z = I
    for i in range(N):
        ZY = Z @ Y
        B = b * ZY + c * ZY @ ZY
        Y = a * Y + Y @ B
        Z = a * Z + B @ Z
        # assert not (torch.isnan(Z).any() or not torch.isinf(Z).any()), 'There is a nan or inf in newtonschulz'
    return Z / torch.sqrt(normalize)

def _matrix_inverse_root_svd(G: torch.Tensor, eps=None, N = None) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(G, full_matrices = False)
    s_clip = torch.clamp(s, min = 0)
    # s_shifted = s - min(s.min().item(), 0) + eps
    return (u @ s_clip.pow_(-0.5).diag() @ vh)

# @torch.compile  # Temporarily disabled - requires Triton installation
def _matrix_inverse_root_PolarExpress(G: torch.Tensor , eps: float,  N: int) -> torch.Tensor: 
    assert G.ndim >= 2 
    coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933), 
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601), 
    (3.9486908534822946 , -2.908902115962949, 0.5518191394370137), 
    (3.3184196573706015 , -2.488488024314874, 0.51004894012372), 
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673), 
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835), 
    (1.8750014808534479 , -1.2500016453999487, 0.3750001645474248), 
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically 
    ]  
    # safety factor for numerical stability (but exclude last polynomial) 
    coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list [:-1]] + [coeffs_list [-1]]
    X = G.float()
    dim = G.size(-1)
    I = torch.eye(dim, dtype=X.dtype, device=X.device)
    # X = G.bfloat16() # for speed
    normalize = (X + eps * I).norm(dim=(-2, -1), keepdim=True) * 1.01
    Y = X / normalize
    Z = I
    hs = coeffs_list[:N] + list(repeat(coeffs_list[-1], N - len(coeffs_list))) 
    for a, b, c in hs: 
        ZY = Z @ Y 
        B = b * ZY + c * ZY @ ZY
        Y = a * Y + Y @ B
        Z = a * Z + B @ Z # X <- aX + bXˆ3 + cXˆ5  
    return Z  / torch.sqrt(normalize)


# @torch.compile  # Temporarily disabled - requires Triton installation
def zeropower_via_PolarExpress(G: torch.Tensor , steps: int) -> torch.Tensor: 
    assert G.ndim >= 2 
    coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933), 
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601), 
    (3.9486908534822946 , -2.908902115962949, 0.5518191394370137), 
    (3.3184196573706015 , -2.488488024314874, 0.51004894012372), 
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673), 
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835), 
    (1.8750014808534479 , -1.2500016453999487, 0.3750001645474248), 
    (1.875, -1.25, 0.375), # subsequent coeffs equal this numerically 
    ]  
    # safety factor for numerical stability (but exclude last polynomial) 
    coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list [:-1]] + [coeffs_list [-1]]
    X = G.float()
    # X = G.bfloat16() # for speed 
    if G.size(-2) > G.size(-1): 
        X = X.mT # this reduces FLOPs 
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01) 
    hs = coeffs_list[:steps] + list(repeat(coeffs_list[-1], steps - len(coeffs_list))) 
    for a, b, c in hs: 
        A = X @ X.mT 
        B=b*A+c*A@A 
        X = a * X + B @ X # X <- aX + bXˆ3 + cXˆ5 
    if G.size(-2) > G.size(-1): X = X.mT 
    return X  


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    # (2,-1.5,0.5)
    # X = G.bfloat16()
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
