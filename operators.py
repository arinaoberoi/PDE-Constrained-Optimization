import numpy as np

def grad(u, hx, hy):
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[1:-1,:] = (u[2:,:] - u[:-2,:])/(2*hx)
    gy[:,1:-1] = (u[:,2:] - u[:,:-2])/(2*hy)
    gx[0,:] = (u[1,:]-u[0,:])/hx
    gx[-1,:] = (u[-1,:]-u[-2,:])/hx
    gy[:,0] = (u[:,1]-u[:,0])/hy
    gy[:,-1] = (u[:,-1]-u[:,-2])/hy
    return gx, gy

def div(vx, vy, hx, hy):
    nx, ny = vx.shape
    out = np.zeros_like(vx)
    out[1:-1,:] += (vx[2:,:]-vx[:-2,:])/(2*hx)
    out[:,1:-1] += (vy[:,2:]-vy[:,:-2])/(2*hy)
    out[0,:] += (vx[1,:]-vx[0,:])/hx
    out[-1,:] += (vx[-1,:]-vx[-2,:])/hx
    out[:,0] += (vy[:,1]-vy[:,0])/hy
    out[:,-1] += (vy[:,-1]-vy[:,-2])/hy
    return out

def laplacian_k(k, hx, hy):
    out = np.zeros_like(k)
    out[1:-1,1:-1] = (
        (k[2:,1:-1] - 2*k[1:-1,1:-1] + k[:-2,1:-1])/(hx*hx)
      + (k[1:-1,2:] - 2*k[1:-1,1:-1] + k[1:-1,:-2])/(hy*hy)
    )
    out[0,1:-1] = (k[1,1:-1]-k[0,1:-1])/(hx*hx) + (k[0,2:]-2*k[0,1:-1]+k[0,:-2])/(hy*hy)
    out[-1,1:-1] = (k[-2,1:-1]-k[-1,1:-1])/(hx*hx) + (k[-1,2:]-2*k[-1,1:-1]+k[-1,:-2])/(hy*hy)
    out[1:-1,0] = (k[2:,0]-2*k[1:-1,0]+k[:-2,0])/(hx*hx) + (k[1:-1,1]-k[1:-1,0])/(hy*hy)
    out[1:-1,-1] = (k[2:,-1]-2*k[1:-1,-1]+k[:-2,-1])/(hx*hx) + (k[1:-1,-2]-k[1:-1,-1])/(hy*hy)
    return out
