# PDE Adjoint Lab
**Inverse Problems & PDE-Constrained Optimization in Python (NumPy-only)**

This mini-engineering project implements a 2D Poisson/anisotropic diffusion forward solver on a regular grid and an **adjoint-state gradient** for **inverse coefficient (conductivity) estimation**. It demonstrates PDE-constrained optimization without automatic differentiation—derivatives are derived and implemented explicitly—exactly the kind of mathematical engineering PhD programs love to see.

## Features
- Finite-difference discretization of `-∇·(k(x) ∇u) = f` on a rectangular domain.
- Dirichlet or homogeneous Neumann boundary conditions.
- Conjugate Gradient (CG) solver for the symmetric positive-definite linear system.
- **Adjoint-state method** to compute gradients of a data-misfit with respect to spatially varying conductivity `k(x)`.
- Tikhonov (H¹-like) regularization with a Laplacian penalty on `k`.
- Barzilai–Borwein gradient descent with backtracking line search.
- Example: recover a “phantom” conductivity from noisy measurements of the state `u` at sensor locations.
- Clean, well-documented NumPy implementation; no autodiff frameworks required.

> Mathematical core (continuous problem):
>
> Find `k(x) ≥ k_min` minimizing  
> J(k) = 1/2 Σ_{i∈S} (u_k(x_i) - d_i)^2 + (α/2) ∫_Ω ||∇k||^2 dx
>
> subject to the PDE:  -∇·(k ∇u) = f  in Ω with boundary conditions.
>
> Adjoint equation:  -∇·(k ∇p) = r  where  r  is a residual with point-sources at sensors.  
> Gradient:  ∂J/∂k = -∇u · ∇p - α Δk.

## Why this is compelling
- Shows mastery of PDEs (elliptic), numerical linear algebra (CG), variational calculus, and inverse problems.
- Implements adjoint derivatives by hand (a standard research technique in PDE-constrained optimization, imaging, and CFD).
- Engineering polish: modular package, tests, docs, and a runnable example.

## Quickstart
```bash
pip install -r requirements.txt
python -m examples.01_inverse_conductivity
```
This will:
1. Build a ground-truth conductivity with inclusions.
2. Solve the forward PDE, sample noisy sensors.
3. Run adjoint-based optimization to reconstruct `k`.
4. Plot ground-truth vs. reconstruction and convergence.

## Repo structure
```
pde-adjoint-lab/
  pdealab/
    __init__.py
    grid.py
    operators.py
    poisson.py
    adjoint.py
    inverse.py
    utils.py
  examples/
    01_inverse_conductivity.py
  requirements.txt
  LICENSE
  README.md
```

## Extending ideas
- Multi-source experiments and multi-frequency Poisson/Helmholtz.
- Total Variation regularization (split Bregman / Chambolle–Pock).
- Shape optimization with level sets (differentiate domain instead of `k`).
- Ultrasound/thermal tomography forward models (time-dependent PDEs).
- Switch linear solver to preconditioned CG (IC/AMG) or multigrid.

## License
MIT
