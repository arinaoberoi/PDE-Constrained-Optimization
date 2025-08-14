import numpy as np
import matplotlib.pyplot as plt

from pdealab import Grid, solve_poisson, InverseProblem
from pdealab import utils

def main():
    g = Grid(nx=96, ny=96, Lx=1.0, Ly=1.0)
    k_true = utils.make_phantom(g, centers=[(0.35,0.35),(0.7,0.65)], radii=[0.12,0.15], k_inside=3.0, k_out=1.0)
    X, Y = g.X, g.Y
    f = np.ones_like(X) * 2.0
    f += 10.0*np.exp(-((X-0.2)**2 + (Y-0.8)**2)/0.01)

    u_true, _ = solve_poisson(f, k_true, g.hx, g.hy)

    sensors, data = utils.sample_sensors(u_true, n_sensors=300, noise=0.01, rng=0)
    k0 = np.ones_like(k_true) * 1.0

    problem = InverseProblem(g, f, k0, sensors, data, kmin=0.5, kmax=5.0, alpha=5e-3)
    k_rec, u_rec, hist = problem.run(maxit=60, tol=1e-7, verbose=True)

    iters = [h[0] for h in hist]
    Jvals = [h[1] for h in hist]

    plt.figure()
    plt.imshow(k_true.T, origin="lower", extent=[0,1,0,1])
    plt.title("True conductivity k(x)")
    plt.colorbar()

    plt.figure()
    plt.imshow(k_rec.T, origin="lower", extent=[0,1,0,1])
    plt.title("Reconstructed conductivity k(x)")
    plt.colorbar()

    plt.figure()
    plt.plot(iters, Jvals)
    plt.xlabel("Iteration")
    plt.ylabel("Objective J")
    plt.title("Convergence")

    plt.show()

if __name__ == "__main__":
    main()
