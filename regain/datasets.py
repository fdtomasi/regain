from sklearn.datasets import make_sparse_spd_matrix
import numpy as np


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def generate(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3):
    theta_tot = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.3, norm_diag=1)

    theta_tot[n_dim_lat:, n_dim_lat:] = make_sparse_spd_matrix(n_dim_obs, alpha=.5, norm_diag=1)
    theta_tot = theta_tot.dot(theta_tot.T)

    assert is_pos_def(theta_tot)
    assert np.linalg.matrix_rank(theta_tot[:n_dim_lat, :n_dim_lat]) == n_dim_lat

    Thetas = [theta_tot]

    threshold = epsilon / np.sqrt(n_dim_lat + n_dim_obs)

    for i in range(2):
        np.random.randn()
        theta_tot_i = make_sparse_spd_matrix(
            n_dim_lat + n_dim_obs, alpha=.95, norm_diag=1)
        theta_tot_i /= np.linalg.norm(theta_tot_i, 'fro')
        theta_tot_i *= epsilon
        theta_tot_i[-threshold < theta_tot_i & theta_tot_i < threshold] = 0
        assert is_pos_def(theta_tot)

generate()
