# Solving linear system in Pytorch
This repository implements most commonly used iterative methods for solving linear system in Pytorch including Conjugate Gradient (CG) and GMRES. 
This implementation can run on GPU and is compatible with both torch.float32 and torch.float64.

![](figs/relative_cg_test_4096.png)

![](figs/relative_gmres_test_2048.png)

## How to use
```python
import torch
from linalg import CG, GMRES

A = torch.tensor([[3.0, 1.0, 0.0], 
                  [1.0, 2.0, -1.0], 
                  [0.0, -1.0, 1.0]])

b = torch.tensor([1.0, 2.0, 3.0])

sol1, info = CG(A, b)
print(f'Solution by CG: {sol1}')

sol2, info = GMRES(A, b)
print(f'Solution by GMRES: {sol2}')

```
Remark: `info` is a tuple where `info[0]` is the number of iterations and `info[1]` is a list of relative residual error at each iteration. 

See more examples in `test.py`. 