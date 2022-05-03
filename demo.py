import torch

from linalg import CG, GMRES

A = torch.tensor([[3.0, 1.0, 0.0], 
                  [1.0, 2.0, -1.0], 
                  [0.0, -1.0, 1.0]])

b = torch.tensor([1.0, 2.0, 3.0])

sol1, _ = CG(A, b)
print(f'Solution by CG: {sol1}')

sol2, _ = GMRES(A, b)
print(f'Solution by GMRES: {sol2}')