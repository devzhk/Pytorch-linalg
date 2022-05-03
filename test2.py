'''
This file tests the accuracy of residual error tracked by CG
'''
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from linalg import CG

#%%
def test(size_sys, init_guess=False):
    mat = np.random.randn(size_sys, size_sys)
    A = mat.T @ mat + np.identity(size_sys)
    b = np.random.randn(size_sys)
    x0 = np.random.randn(size_sys) if init_guess else None

    A = torch.from_numpy(A)#.to(torch.float32)
    b = torch.from_numpy(b)#.to(torch.float32)
    x0 = torch.from_numpy(x0)#.to(torch.float32)
    # x0 = None

    sol, (num_iter, err_list) = CG(A, b, x0, track_res=True)
    res_gt = torch.norm(b - A @ sol)
    bnorm = torch.norm(b)
    rel_err = res_gt / bnorm
    print(f'Relative error: {rel_err}')
    return np.sqrt(np.array(err_list)) / bnorm

#%%
torch.set_default_dtype(torch.float64)

errs = test(512, True)
# %%
xs = list(range(len(errs)))

plt.plot(xs, errs)
plt.xlabel('Steps')
plt.ylabel('Relative error')
plt.yscale('log')
plt.show()
# %%
print(errs[-1])
# %%
