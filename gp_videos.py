import numpy as np 
# multiprocessing 
from multiprocessing import Pool
from itertools import product
from functools import partial

N_x = 30
N_y = 30
N_t = 20 
Ls = [0.5,1,2,4,8,16,32]
Ts = [0.5,1,2,4,8,16,32]

def kernel(x1, x2=None, L=1, T=1):
    """x1 and x2 are arrays of 3D points (x, y, t) of shape (n, 3).ipynb_checkpoints/
    Returns the kernel matrix of the data points of shape (n, n) where the (i, j)th element is the kernel function evaluated at x1[i] and x2[j]."""
    if x2 is None:
        x2 = x1
    x1_d = x1[:,:2] # N,2 
    x2_d = x2[:,:2] # N,2
    x1_t = x1[:,2] # N
    x2_t = x2[:,2] # N
    d_dist = np.linalg.norm(x1_d[:,np.newaxis,:] - x2_d[np.newaxis,:,:],axis=2) # N,N
    t_dist = np.abs(x1_t[:,np.newaxis] - x2_t[np.newaxis,:]) # N,N    
    exp_d = np.exp(- d_dist**2 / (2*(L**2))) # N,N
    exp_t = np.exp(- t_dist**2 / (2*(T**2))) # N,N

    return exp_d * exp_t

def sample_video(L, T, N_x=10, N_y=10, N_t=5):
    # Get coordinates of 3D space-time points
    x = np.arange(N_x)
    y = np.arange(N_y)
    t = np.arange(N_t)
    X_, Y_, T_ = np.meshgrid(x, y, t)
    coords = np.array([X_,Y_,T_])
    coords_shape = (N_x, N_y, N_t)
    coords = coords.reshape(3,-1) # flatten into 2D
    # Calculate kernel matrix
    K = kernel(coords.T, L=L, T=T)
    
    # Sample from N(0,K) using Cholesky decomposition
    L = np.linalg.cholesky(K + 1e-8*np.eye(K.shape[0]))
    N = np.random.normal(size=(K.shape[0]))
    V = np.dot(L, N)
    V = V.reshape(coords_shape)
    return V

if __name__ == "__main__":

    # Get inputs from command line 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1.0)
    parser.add_argument("--T", type=int, default=1.0)
    parser.add_argument("--i", type=int, default=10)
    parser.add_argument("--dt", type=str, default=10)

    args = parser.parse_args()
    L = Ls[args.L]
    T = Ts[args.T]
    
    results = sample_video(
        L=L,
        T=T,
        N_x=N_x,
        N_y=N_y,
        N_t=N_t
    )
    random_hash = str(np.random.randint(1e8))
    np.savez(f"results/{args.dt}/{random_hash}.npz", results=results, L_idx=args.L, T_idx=args.T, id=args.i)