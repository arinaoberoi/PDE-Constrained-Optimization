import numpy as np

def make_phantom(grid, centers, radii, k_inside=3.0, k_out=1.0):
    k = k_out * np.ones(grid.shape)
    for (cx, cy), r in zip(centers, radii):
        mask = (grid.X - cx)**2 + (grid.Y - cy)**2 <= r**2
        k[mask] = k_inside
    return k

def sample_sensors(u, n_sensors=200, noise=0.01, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    nx, ny = u.shape
    idx = rng.integers(0, nx, size=n_sensors)
    idy = rng.integers(0, ny, size=n_sensors)
    sensors = list(zip(idx.tolist(), idy.tolist()))
    data = np.array([u[i,j] for i,j in sensors], dtype=float)
    data += noise * rng.standard_normal(size=data.shape)
    return sensors, data
