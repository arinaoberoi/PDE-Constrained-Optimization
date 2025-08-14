import numpy as np
from .operators import grad, div

def A_apply(u, k, hx, hy):
    ux, uy = grad(u, hx, hy)
    jx = k * ux
    jy = k * uy
    return -div(jx, jy, hx, hy)

def cg(A, b, x0=None, tol=1e-8, maxit=500):
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x)
    p = r.copy()
    rs = np.dot(r.ravel(), r.ravel())
    for it in range(maxit):
        Ap = A(p)
        denom = np.dot(p.ravel(), Ap.ravel())
        if abs(denom) < 1e-30:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r.ravel(), r.ravel())
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / max(1e-30, rs)
        p = r + beta * p
        rs = rs_new
    return x, it+1

def solve_poisson(f, k, hx, hy, u0=None, tol=1e-8, maxit=500):
    A = lambda x: A_apply(x.reshape(f.shape), k, hx, hy).ravel()
    b = f.ravel()
    x0 = None if u0 is None else u0.ravel()
    sol, it = cg(A, b, x0=x0, tol=tol, maxit=maxit)
    return sol.reshape(f.shape), it
