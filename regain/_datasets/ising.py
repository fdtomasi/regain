import warnings
import numpy as np

from regain.generalized_linear_model.sampling import ising_sampler

#TODO UPDATE ISING DIFFUSION PROCESSES

# TODO update with switch of an edge

def update_ising_l1(theta_init, no, n_dim_obs, responses=[-1,1]):
    theta = theta_init.copy()
    rows = np.zeros(no)
    cols = np.zeros(no)
    while (np.any(rows == cols)):
        rows = np.random.randint(0, n_dim_obs, no)
        cols = np.random.randint(0, n_dim_obs, no)
        for r, c in zip(rows, cols):
            theta[r, c] = np.random.choice(responses+[0]) \
                          if theta[r, c] == 0 \
                          else np.random.choice([theta[r,c], 0])  # np.random.rand(1) * .35
            theta[c, r] = theta[r, c]
    assert (np.all(theta==theta.T))
    return theta

def ising_theta_generator(p=10, n=100, T=10, mode='l1', time_on_axis='first',
          change=4, responses=[-1,1]):
    theta = np.random.choice(np.array([-0.5, 0, 0.5]), size=(p,p), replace=True)
    theta += theta.T
    np.fill_diagonal(theta, 0)
    theta[np.where(theta>0)] = 1
    theta[np.where(theta<0)] = -1
    thetas = [theta]


    for t in range(1, T):
        if mode == 'switch':
            raise ValueError("Still not implemented")
        elif mode== 'diffusion':
            raise ValueError("Still not implemented")
        elif mode == 'l1':
            theta_t = update_ising_l1(thetas[-1], change, theta.shape[0])
        else:
            warnings.warn("Mode not implemented. Using l1.")
            theta_t = update_ising_l1(thetas[-1], change, theta.shape[0])
        thetas.append(theta_t)
    return thetas
