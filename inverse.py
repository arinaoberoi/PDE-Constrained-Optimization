import numpy as np
from .poisson import solve_poisson
from .adjoint import adjoint_state, misfit_gradient, sensor_residual

def barzilai_borwein(g_prev, g_curr, s_prev):
    y = g_curr - g_prev
    denom = np.dot(y.ravel(), y.ravel())
    if denom < 1e-30:
        return 1e-2
    return max(1e-6, min(1e2, np.dot(s_prev.ravel(), y.ravel()) / denom))

class InverseProblem:
    def __init__(self, grid, f, k0, sensors, data, kmin=0.1, kmax=5.0, alpha=1e-2):
        self.grid = grid
        self.f = f
        self.k = k0.copy()
        self.sensors = sensors
        self.data = data
        self.kmin = kmin
        self.kmax = kmax
        self.alpha = alpha
        self.history = []

    def project(self, k):
        return np.clip(k, self.kmin, self.kmax)

    def objective(self, u):
        return sensor_residual(u, self.sensors, self.data)

    def solve_forward(self, k):
        u, _ = solve_poisson(self.f, k, self.grid.hx, self.grid.hy)
        return u

    def step(self, k, gk, step):
        tau = step
        u = self.solve_forward(k)
        J0 = self.objective(u)
        while tau > 1e-8:
            k_new = self.project(k - tau * gk)
            u_new = self.solve_forward(k_new)
            J_new = self.objective(u_new)
            if J_new <= J0 - 1e-4 * tau * np.dot(gk.ravel(), gk.ravel()):
                return k_new, u_new, J_new, tau
            tau *= 0.5
        return k, u, J0, 0.0

    def run(self, maxit=50, tol=1e-6, verbose=True):
        k = self.k.copy()
        u = self.solve_forward(k)
        J = self.objective(u)
        g_prev = None
        s_prev = None
        for it in range(1, maxit+1):
            p = adjoint_state(u, k, self.grid.hx, self.grid.hy, self.sensors, self.data)
            g = misfit_gradient(u, p, k, self.grid.hx, self.grid.hy, alpha_reg=self.alpha)
            step = 1e-2 if g_prev is None else barzilai_borwein(g_prev, g, s_prev)
            k_new, u_new, J_new, tau = self.step(k, g, step)
            self.history.append((it, J_new, tau, np.linalg.norm(g)))
            if verbose:
                print(f"iter {it:3d} | J={J_new:.6e} | step={tau:.2e} | ||g||={np.linalg.norm(g):.3e}")
            if abs(J - J_new) < tol:
                k, u, J = k_new, u_new, J_new
                break
            s_prev = k_new - k
            g_prev = g.copy()
            k, u, J = k_new, u_new, J_new
        self.k = k
        return k, u, self.history
