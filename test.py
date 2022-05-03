from tqdm import tqdm

import numpy as np
from scipy.sparse.linalg import cg as sp_cg
from scipy.sparse.linalg import gmres as sp_gmres
from jax.scipy.sparse.linalg import cg as jx_cg
from jax.scipy.sparse.linalg import gmres as jax_gmres

import matplotlib.pyplot as plt
import torch
from functools import partial
from linalg import CG, GMRES


def Mvp(A, vec):
    return A @ vec


def test_cg(A, b, maxiter, x0=None):
    th_A = torch.from_numpy(A)
    th_b = torch.from_numpy(b)
    if x0 is not None:
        th_x0 = torch.from_numpy(x0)
    else:
        th_x0 = None
    LinOp = partial(Mvp, th_A)

    sp_sol = sp_cg(A, b, x0=x0, maxiter=maxiter)
    # print(sp_sol)
    res_sp = np.linalg.norm(A @ sp_sol[0] - b)

    jx_sol = jx_cg(A, b, x0=x0, maxiter=maxiter)
    # print(jx_sol)
    res_jx = np.linalg.norm(A @ sp_sol[0] - b)

    sol,_ = CG(LinOp, th_b, x0=th_x0, tol=1e-5, max_iter=maxiter)
    # print(sol)
    res_th = torch.norm(LinOp(sol) - th_b).item()
    return res_sp, res_jx, res_th


def test_gmres(A, b, maxiter, x0=None):
    th_A = torch.from_numpy(A)
    th_b = torch.from_numpy(b)
    if x0 is not None:
        th_x0 = torch.from_numpy(x0)
    else:
        th_x0 = None
    LinOp = partial(Mvp, th_A)

    sp_sol = sp_gmres(A, b, x0=x0, restart=maxiter, maxiter=1)
    # print(sp_sol)
    res_sp = np.linalg.norm(A @ sp_sol[0] - b)

    jx_sol = jax_gmres(A, b, x0=x0, restart=maxiter, maxiter=1)
    # print(jx_sol)
    res_jx = np.linalg.norm(A @ sp_sol[0] - b)

    sol, _ = GMRES(LinOp, th_b, x0=th_x0, tol=1e-5, max_iter=maxiter)
    # print(sol)
    res_th = torch.norm(LinOp(sol) - th_b).item()
    return res_sp, res_jx, res_th



def plot_test(size_sys, algo='cg', err_type='abs', init_guess=False):
    dtype = [np.float64, np.float32]

    K = int(np.log2(size_sys)) + 1

    mat = np.random.randn(size_sys, size_sys)
    A = mat.T @ mat + np.identity(size_sys)
    b = np.random.randn(size_sys)
    x0 = np.random.randn(size_sys) if init_guess else None

    iter_list = [2 ** k for k in range(K)]

    for dt in dtype:
        if dt == np.float32:
            A = A.astype(dt)
            b = b.astype(dt)
            if x0 is not None:
                x0 = x0.astype(dt)
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float64)
        sp_list = []
        jx_list = []
        th_list = []
        # float64
        for k in iter_list:
            maxiter = k
            if algo == 'cg':
                res_sp, res_jx, res_th = test_cg(A, b, maxiter, x0)
            elif algo == 'gmres':
                res_sp, res_jx, res_th = test_gmres(A, b, maxiter, x0)
            else:
                raise ValueError(f'{algo} not supported')
            if err_type == 'relative':
                bnorm = np.linalg.norm(b)
                res_sp = res_sp / bnorm
                res_jx = res_jx / bnorm
                res_th = res_th / bnorm

            sp_list.append(res_sp)
            jx_list.append(res_jx)
            th_list.append(res_th)
        line, = plt.plot(iter_list, sp_list, label=f'scipy {algo}-{dt}', alpha=0.5, marker='*')
        line1, = plt.plot(iter_list, jx_list, label=f'jax {algo}-{dt}', alpha=0.5, marker='+')
        line2, = plt.plot(iter_list, th_list, label=f'torch {algo}-{dt}', alpha=0.5, marker='o')
    plt.legend()
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel(f'L2 error ({err_type})')
    plt.savefig(f'figs/{err_type}_{algo}_test_{size_sys}.png')
    plt.cla()


if __name__ == '__main__':
    sizes = [128, 256, 512, 1024, 2048]
    algo = 'cg'
    err_type = 'relative'

    for size_sys in tqdm(sizes):
        plot_test(size_sys, algo, err_type, init_guess=False)
