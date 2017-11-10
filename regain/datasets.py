"""Dataset generation module."""
import numpy as np
import sys
import scipy.sparse as ss

from sklearn.datasets import make_sparse_spd_matrix


def is_pos_def(x):
    """Check if x is positive-definite."""
    return np.all(np.linalg.eigvals(x) > 0)
def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def generate(n_dim_obs=3, n_dim_lat=2, eps=1e-3, T=10, tol=1e-3):
    theta = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.3, norm_diag=1)

    theta[n_dim_lat:, n_dim_lat:] = make_sparse_spd_matrix(
        n_dim_obs, alpha=.5, norm_diag=1)

#     theta_tot = theta_tot.dot(theta_tot.T)
#     assert is_pos_def(theta_tot)
    assert np.linalg.matrix_rank(theta[:n_dim_lat, :n_dim_lat]) == n_dim_lat

    thetas = [theta]
    for i in range(T):
        theta_tot_i = make_sparse_spd_matrix(
            n_dim_lat + n_dim_obs, alpha=.5, norm_diag=1)
        theta_tot_i.flat[::n_dim_lat + n_dim_obs + 1] = 1e-16
        theta_tot_i /= np.linalg.norm(theta_tot_i, 'fro')
        theta_tot_i *= eps

        theta = thetas[-1].copy()
        theta += theta_tot_i
        theta.flat[::n_dim_lat + n_dim_obs + 1] = 0
        theta /= np.amax(np.abs(theta))
        # min((np.linalg.norm(theta2, 'fro')/((n_dim_lat + n_dim_obs)**2)), 0.99)
        threshold = eps / np.power(n_dim_lat + n_dim_obs, 2)
        theta[np.abs(theta) < threshold] = 0
        theta[np.abs(theta) < tol] = 0
        theta.flat[::n_dim_lat + n_dim_obs + 1] = 1
        # assert is_pos_def(theta)
        thetas.append(theta)
    return np.array(thetas)


def generate2(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3, T=10):
    theta_tot = make_sparse_spd_matrix(n_dim_lat + n_dim_obs, alpha=.3, norm_diag=1)
    theta_tot[0:n_dim_lat, 0:n_dim_lat] = np.diag(np.ones(n_dim_lat))
    theta_tot[n_dim_lat:, n_dim_lat:] = make_sparse_spd_matrix(
        n_dim_obs, alpha=.5, norm_diag=1)
    #print(theta_tot)
#     theta_tot = theta_tot.dot(theta_tot.T)
#     assert is_pos_def(theta_tot)
    assert np.linalg.matrix_rank(theta_tot[:n_dim_lat, :n_dim_lat]) == n_dim_lat

    L = theta_tot[n_dim_lat:, 0:n_dim_lat].dot(
          theta_tot[0:n_dim_lat, n_dim_lat:])
    assert (is_pos_semidef(L))
    thetas = [theta_tot]
    thetas_true = [theta_tot[n_dim_lat:, n_dim_lat:]]
    thetas_observed = [thetas_true[-1] - L]
    assert (is_pos_def(thetas_observed[-1]))

    if(not is_pos_def(thetas_observed[-1])):
        sys.exit(0)
    i = 0
    while i < T:
      u, s, v = np.linalg.svd(thetas[-1][n_dim_lat:, n_dim_lat:])
      addition = ss.rand(s.shape[0],1, density=0.5).A
      addition =  addition / (np.linalg.norm(addition) * np.sqrt(epsilon))

      theta = np.zeros_like(thetas[-1])
      s_new = (s + addition.T)[0]
      print(s_new)
      s_new = np.maximum(s_new, 0)
      theta[n_dim_lat:, n_dim_lat:] = u.dot(np.diag(s_new)).dot(v.T)

      print(thetas[-1][n_dim_lat:, 0:n_dim_lat].shape)
      u, s, v = np.linalg.svd(thetas[-1][n_dim_lat:, 0:n_dim_lat])

      addition = ss.rand(s.shape[0], 1, density=0.5).A
      addition =  addition / (np.linalg.norm(addition) * np.sqrt(epsilon))
      print(s.shape)
      print(addition.shape)
      s_new = (np.square(s) + addition.T)[0]
      s_new = np.maximum(s_new, 0)

      print("s_new shape", s_new.shape)
      print("u shape", u.shape)
      print("v shape", v.shape)

      theta[n_dim_lat:, 0:n_dim_lat] = u.dot(
        np.concatenate((np.diag(np.sqrt(s_new)), np.zeros((1,3))), axis=0)).dot(v.T)
      theta[0:n_dim_lat, n_dim_lat:] = theta[n_dim_lat:, 0:n_dim_lat].T
      L = theta_tot[n_dim_lat:, 0:n_dim_lat].dot(
          theta_tot[0:n_dim_lat, n_dim_lat:])

      thetas.append(theta)
      thetas_true.append(theta[n_dim_lat:, n_dim_lat:])
      thetas_observed.append(thetas_true[-1] - L)

      assert(is_pos_semidef(L))
      assert(is_pos_def(thetas_observed[-1]))
      print(theta)
      i+=1
    return thetas, thetas_true, thetas_observed




def generate_dataset(n_dim_obs=3, n_dim_lat=2, epsilon=1e-3, T=10,
                     n_samples = 50):
    thetas = generate(n_dim_obs, n_dim_lat, epsilon, T)
def generate_dataset(n_dim_obs=3, n_dim_lat=2, eps=1e-3, T=10,
                     n_samples=50):
    thetas = generate(n_dim_obs, n_dim_lat, eps, T)
    sigmas = np.array(map(np.linalg.inv, thetas))
    data_list = []
    for sigma in sigmas:
        data_list.append(np.random.multivariate_normal(
            np.zeros(n_dim_obs), sigma[n_dim_lat:, n_dim_lat:], n_samples))
    return data_list, thetas, np.array([t[n_dim_lat:, n_dim_lat:] for t in thetas])


def generate_ma_xue_zou(n_dim_obs=12, n_dim_lat=2, epsilon=1e-3):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
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
        L = K[n_dim_lat:, 0:n_dim_lat].dot(K[0:n_dim_lat, 0:n_dim_lat]).dot(
            K[0:n_dim_lat, n_dim_lat:])
        assert (is_pos_semidef(L))
        assert (is_pos_def(K[n_dim_lat:, n_dim_lat:] - L))

    except:
        sys.stdout.write("-")
