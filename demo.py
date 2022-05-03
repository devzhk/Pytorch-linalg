import torch
from linalg import CG, GMRES
from functools import partial

A = torch.tensor([[3.0, 1.0, 0.0], 
                  [1.0, 2.0, -1.0], 
                  [0.0, -1.0, 1.0]])

b = torch.tensor([1.0, 2.0, 3.0])

sol1, _ = CG(A, b)
print(f'Solution by CG: {sol1}')

sol2, _ = GMRES(A, b)
print(f'Solution by GMRES: {sol2}')


def Avp(A, vec):
    return A @ vec

# create custom linear operator that produces Ax
LinOp = partial(Avp, A)

sol3, _ = CG(LinOp, b)
print(f'Solution by CG: {sol3}')

sol4, _ = GMRES(LinOp, b)
print(f'Solution by GMRES: {sol4}')