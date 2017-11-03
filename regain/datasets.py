from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
import sys

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def generate(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3, T=10):
    theta_tot = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.3, norm_diag=1)

    theta_tot[n_dim_lat:, n_dim_lat:] = make_sparse_spd_matrix(
        n_dim_obs, alpha=.5, norm_diag=1)

#     theta_tot = theta_tot.dot(theta_tot.T)
#     assert is_pos_def(theta_tot)
    assert np.linalg.matrix_rank(theta_tot[:n_dim_lat, :n_dim_lat]) == n_dim_lat

    thetas = [theta_tot]
    for i in range(T):
        theta_tot_i = make_sparse_spd_matrix(
            n_dim_lat + n_dim_obs, alpha=.5, norm_diag=1)
        theta_tot_i.flat[::n_dim_lat + n_dim_obs + 1] = 1e-16
        theta_tot_i /= np.linalg.norm(theta_tot_i, 'fro')
        theta_tot_i *= epsilon

        theta = thetas[-1].copy()
        theta += theta_tot_i
        theta.flat[::n_dim_lat + n_dim_obs + 1] = 0
        theta /= np.amax(np.abs(theta))
        # min((np.linalg.norm(theta2, 'fro')/((n_dim_lat + n_dim_obs)**2)), 0.99)
        threshold =epsilon/((n_dim_lat + n_dim_obs)**2)
        theta[np.logical_and(theta < threshold, -threshold < theta)] = 0
        theta.flat[::n_dim_lat + n_dim_obs + 1] = 1
        thetas.append(theta)
    return thetas


def generate_dataset(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3, T=10,
                     n_samples = 50):
    thetas = generate(n_dim_obs, n_dim_lat, epsilon, T)
    sigmas = np.array(map(np.linalg.inv, thetas))
    data_list = []
    for sigma in sigmas:
        data_list.append(np.random.multivariate_normal(np.zeros(sigma.shape[0]),
                                                       sigma,
                                                       n_samples)))
    return data_list


def generate_ma_xue_zou(n_dim_obs=12, n_dim_lat=2, epsilon=1e-3):
    theta = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.9, norm_diag=1)
    theta_flat = theta.flatten()
    idx = (theta_flat != 0)

    proba = np.random.randn(*theta_flat.shape)
    proba = np.where(proba > 0, 1, -1)

    theta_flat[idx] = proba[idx]

    U = theta_flat.reshape(theta.shape)
    UTU = U.dot(U.T)
#     UTU.flat[::theta.shape[0]+1] = 1
    try:
        K = np.linalg.inv(UTU)
        assert is_pos_def(K)
    except:
        sys.stdout.write("-")
