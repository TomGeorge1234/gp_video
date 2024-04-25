import numpy as np 
import scipy

N_x = 36 # no. position bins (x)
N_y = 36 # no. position bins (y)
N_t = 60 # no. time bins
Ls = [0.5,1,2,4,8,16,32] # decoherence length scale (pixels)
Ts = [0.5,1,2,4,8,16,32] # decoherence time scale (frames)

def kernel(x1, x2=None, L=1, T=1):
    """x1 and x2 are arrays of 3D points [x, y, t] of shape (n, 3). 
    Returns the kernel matrix of the data points of shape (n, n) where the (i, j)th element is the kernel function evaluated at x1[i] and x2[j].
    The kernel is a product of a spatial and temporal Gaussian kernel of sigmas L and T respectively."""
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


def sample_video(L, T, N_x=36, N_y=36, N_t=60):
    """Samples a video of shape (N_x, N_y, N_t) from a Gaussian process."""

    # coordinates of the final video
    x = np.arange(N_x)
    y = np.arange(N_y)
    t = np.arange(N_t)
    X_, Y_, T_ = np.meshgrid(x, y, t)

    if (L < 1) or (T < 1):
        # In this case, pixels / frames are independent and we can sample them separately, lowering the memory cost of sampling the covariance matrix
        V = np.zeros((N_x, N_y, N_t))  
        if L < 1: 
            # sample each pixel independently
            for px in range(N_x):
                for py in range(N_y):
                    x = np.array([px]*N_t)
                    y = np.array([py]*N_t)
                    t = np.arange(N_t)
                    coords = np.array([x,y,t]) # (3,N_t)
                    K = kernel(coords.T, L=L, T=T)
                    L_ = np.linalg.cholesky(K + 1e-8*np.eye(K.shape[0]))
                    N = np.random.normal(size=(K.shape[0])) 
                    V_ = np.dot(L_, N)
                    V[px,py,:] = V_    
        elif T < 1: 
            for ft in range(N_t):
                x = np.arange(N_x)
                y = np.arange(N_y)
                x_, y_, t = np.meshgrid(x, y, ft)
                coords = np.array([x_,y_,t])
                coords_shape = (N_x, N_y)
                coords = coords.reshape(3,-1)
                K = kernel(coords.T, L=L, T=T)
                L_ = np.linalg.cholesky(K + 1e-8*np.eye(K.shape[0]))
                N = np.random.normal(size=(K.shape[0]))
                V_ = np.dot(L_, N)
                V_ = V_.reshape(coords_shape)
                V[:,:,ft] = V_

    else: 
        coords = np.array([X_,Y_,T_])
        coords_shape = (N_x, N_y, N_t)
        coords = coords.reshape(3,-1) # flatten into 2D
        # In this case you cant make the independence assumption but you can sample at half resolution and then interpolate.
        # half coordinates (every other point), well interpolate up to full resolution later
        xh = np.arange(N_x + (N_x%2==0))[::2]; N_xh = len(xh)
        yh = np.arange(N_y + (N_y%2==0))[::2]; N_yh = len(yh)
        th = np.arange(N_t + (N_t%2==0))[::2]; N_th = len(th)
        Xh_, Yh_, Th_ = np.meshgrid(xh, yh, th)
        coordsh = np.array([Xh_,Yh_,Th_])
        coordsh_shape = (N_xh, N_yh, N_th)
        coordsh = coordsh.reshape(3,-1) # flatten into 2D
        
        # Calculate kernel matrix
        K = kernel(coordsh.T, L=L, T=T)

        # To sample from a Gaussian N(0,K) is very time consuming. It turns out if you can calculte the cholesky decomposition of K = LL^T, then you can sample from N(0,K) by computing N(0,I) and multiplying by L. see https://rinterested.github.io/statistics/multivariate_normal_draws
        L_ = np.linalg.cholesky(K + 1e-8*np.eye(K.shape[0])) # this is the expensive step, borderline possible for dim(K) < 20,000, after this...good luck! The problem is memory.
        N = np.random.normal(size=(K.shape[0]))
        V = np.dot(L_, N)
        V = V.reshape(coordsh_shape)

        # interpolate up to full resolution
        V = scipy.interpolate.interpn((xh, yh, th), V, coords.T, method='linear', bounds_error=False, fill_value=0)
        V = V.reshape(coords_shape)
    
    return V

if __name__ == "__main__":

    # Get L, T, i and dt from command line arguments
    # L and T are indices for the Ls and Ts arrays, selecting the decoherence length and time scales
    # i is a repeat index. It's not used but will be saved with the output so you can remember this is the i'th run of the script
    # dt is the date and time of the script run, used to find the folder where to save the output
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1.0)
    parser.add_argument("--T", type=int, default=1.0)
    parser.add_argument("--i", type=int, default=10)
    parser.add_argument("--dt", type=str, default=10)
    args = parser.parse_args()
    
    results = sample_video(
        L=Ls[args.L],
        T=Ts[args.T],
        N_x=N_x,
        N_y=N_y,
        N_t=N_t
    )
    random_hash = str(np.random.randint(1e8)) # save the results with a random hash to avoid overwriting
    np.savez(f"results/{args.dt}/{random_hash}.npz", results=results, L_idx=args.L, T_idx=args.T, id=args.i)