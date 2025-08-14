import numpy as np
from .poisson import solve_poisson
from .operators import grad, laplacian_k

def sensor_residual(u, sensors, data):
    r = 0.0
    for (ix, iy), d in zip(sensors, data):
        r += 0.5*(u[ix,iy] - d)**2
    return r

def sensor_rhs(u, sensors, data, shape):
    rhs = np.zeros(shape)
    for (ix, iy), d in zip(sensors, data):
        rhs[ix, iy] += (u[ix,iy] - d)
    return rhs

def adjoint_state(u, k, hx, hy, sensors, data):
    rhs = sensor_rhs(u, sensors, data, u.shape)
    p, _ = solve_poisson(rhs, k, hx, hy)
    return p

def misfit_gradient(u, p, k, hx, hy, alpha_reg=1e-2):
    ux, uy = grad(u, hx, hy)
    px, py = grad(p, hx, hy)
    inner = -(ux*px + uy*py)
    reg = -alpha_reg * laplacian_k(k, hx, hy)
    return inner + reg
