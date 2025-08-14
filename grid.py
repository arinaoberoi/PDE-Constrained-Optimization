import numpy as np

class Grid:
    def __init__(self, nx=128, ny=128, Lx=1.0, Ly=1.0):
        self.nx, self.ny = int(nx), int(ny)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.hx = self.Lx / (self.nx - 1)
        self.hy = self.Ly / (self.ny - 1)
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")

    @property
    def shape(self):
        return (self.nx, self.ny)
