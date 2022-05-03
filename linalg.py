import torch
from functools import partial


def _check_nan(vec, msg):
    if torch.isnan(vec).any():
        raise ValueError(msg)


def _safe_normalize(x, threshold=None):
    norm = torch.norm(x)
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm


def Mvp(A, vec):
    return A @ vec


def arnoldi(vec,    # Matrix vector product
            V,      # List of existing basis
            H,      # H matrix
            j):     # number of basis
    '''
    Arnoldi iteration to find the j th l2-orthonormal vector
    compute the j-1 th column of Hessenberg matrix
    '''
    _check_nan(vec, 'Matrix vector product is Nan')

    for i in range(j):
        H[i, j - 1] = torch.dot(vec, V[i])
        vec = vec - H[i, j-1] * V[i]
    new_v, vnorm = _safe_normalize(vec)
    H[j, j - 1] = vnorm
    return new_v


def cal_rotation(a, b):
    '''
    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}
    '''
    c = torch.sqrt(a * a + b * b)
    return a / c, - b / c


def apply_given_rotation(H, cs, ss, j):
    '''
    Apply givens rotation to H columns
    :param H:
    :param cs:
    :param ss:
    :param j:
    :return:
    '''
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * H[i, j] - ss[i] * H[i + 1, j]
        H[i + 1, j] = cs[i] * H[i+1, j] + ss[i] * H[i, j]
        H[i, j] = tmp
    cs[j], ss[j] = cal_rotation(H[j, j], H[j + 1, j])
    H[j, j] = cs[j] * H[j, j] - ss[j] * H[j + 1, j]
    H[j + 1, j] = 0
    return H, cs, ss


'''
    GMRES solver for solving Ax=b. 
    Reference: https://web.stanford.edu/class/cme324/saad-schultz.pdf
'''

def GMRES(A,                # Linear operator, matrix or function
          b,                # RHS of the linear system in which the first half has the same shape as grad_gx, the second half has the same shape as grad_fy
          x0=None,          # initial guess, tuple has the same shape as b
          max_iter=None,    # maximum number of GMRES iterations
          tol=1e-6,         # relative tolerance
          atol=1e-6,        # absolute tolerance
          track=False):     # If True, track the residual error of each iteration
    '''
    Return:
        sol: solution
        (j, err_history): 
            j is the number of iterations used to achieve the target accuracy; 
            err_history is a list of relative residual error at each iteration if track=True, empty list otherwise. 
    '''  
    if isinstance(A, torch.Tensor):
        Avp = partial(Mvp, A)
    elif hasattr(A, '__call__'):
        Avp = A
    else:
        raise ValueError('A must be a function or matrix')

    bnorm = torch.norm(b)

    if max_iter == 0 or bnorm < 1e-8:
        return b

    if max_iter is None:
        max_iter = b.shape[0]

    if x0 is None:
        x0 = torch.zeros_like(b)
        r0 = b
    else:
        r0 = b - Avp(x0)

    new_v, rnorm = _safe_normalize(r0)
    # initial guess residual
    beta = torch.zeros(max_iter + 1, device=b.device)
    beta[0] = rnorm
    err_history = []
    if track:
        err_history.append((rnorm / bnorm).item())

    V = []
    V.append(new_v)
    H = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
    cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
    ss = torch.zeros(max_iter, device=b.device)  # sine values at each step

    for j in range(max_iter):
        p = Avp(V[j])
        new_v = arnoldi(p, V, H, j + 1)  # Arnoldi iteration to get the j+1 th basis
        V.append(new_v)

        H, cs, ss = apply_given_rotation(H, cs, ss, j)
        _check_nan(cs, f'{j}-th cosine contains NaN')
        _check_nan(ss, f'{j}-th sine contains NaN')
        beta[j + 1] = ss[j] * beta[j]
        beta[j] = cs[j] * beta[j]
        residual = torch.abs(beta[j + 1])
        if track:
            err_history.append((residual / bnorm).item())
        if residual < tol * bnorm or residual < atol:
            break
    y, _ = torch.triangular_solve(beta[0:j + 1].unsqueeze(-1), H[0:j + 1, 0:j + 1])  # j x j
    V = torch.stack(V[:-1], dim=0)
    sol = x0 + V.T @ y.squeeze(-1)
    return sol, (j, err_history)


'''
  Conjugate Gradient algorithm for solving Ax=b. 
  Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method
'''
  
def CG(A,                   # linear operator
       b,                   # RHS of the linear system
       x0=None,             # initial guess
       max_iter=None,       # maximum number of iterations
       tol=1e-5,            # relative tolerance
       atol=1e-6,           # absolute tolerance
       track=False,         # if True, track the residual error of each iteration
       ):
    '''
    Return:
        sol: solution
        (j, err_history): 
            j is the number of iterations used to achieve the target accuracy; 
            err_history is a list of relative residual error at each iteration if track=True, empty list otherwise. 
    '''  
    if isinstance(A, torch.Tensor):
        Avp = partial(Mvp, A)
    elif hasattr(A, '__call__'):
        Avp = A
    else:
        raise ValueError('A must be a function or squared matrix')

    if max_iter is None:
        max_iter = b.shape[0]
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.detach().clone()
    else:
        Av = Avp(x0)
        r = b.detach().clone() - Av
        x = x0

    p = r.clone()
    rdotr = torch.dot(r, r)
    err_history = []
    if track:
        err_history.append(rdotr.item())

    residual_tol = max(tol * tol * torch.dot(b, b), atol * atol)
    if rdotr < residual_tol:
        return x, 0

    for i in range(max_iter):
        Ap = Avp(p)

        alpha = rdotr / torch.dot(p, Ap)
        x.add_(alpha * p)
        r.add_(-alpha * Ap)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if track:
            err_history.append(rdotr.item())
        if rdotr < residual_tol:
            break
    return x, (i + 1, err_history)