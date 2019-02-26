import numpy as np
from scipy import linalg
from sklearn.utils.extmath import fast_logdet


def h(D, K, delta=5):
    return - 0.5 * (np.trace(K.T.dot(D)) - (delta - 2) * fast_logdet(K))


def first_derivative_h(D, K, delta=5):
    return - 0.5 * (D - (delta - 2) * linalg.pinvh(K))


# def second_derivative_h(D, K, delta=5):
#     K_inv = linalg.pinvh(K)
#     return - 0.5 * (delta - 2) * K_inv.dot(K_inv)


def first_derivative_h_version2(D, K, delta=5):
    res = np.empty_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ones = np.zeros_like(D)
            ones[i, j] = ones[j, i] = 1.
            res[i, j] = np.trace((D - (delta - 2) * linalg.pinvh(K)).dot(ones))
    return - res / 2.


def second_derivative_h_version2(D, K, delta=5):
    K_inv = linalg.pinvh(K)
    num_edges = np.triu(K != 0, 1).sum()
    dof = num_edges + K.shape[0]
    H = np.empty((dof, dof))
    # H2 = np.empty((dof, dof))
    Vi, Vj = np.nonzero(np.triu(K != 0))

    # to be the same as matlab
    idx = np.argsort(Vj)
    Vi, Vj = Vi[idx], Vj[idx]

    for ei in range(dof):
        i, j = Vi[ei], Vj[ei]

        # ones1 = np.zeros_like(K)
        # ones1[i, j] = ones1[j, i] = 1.
        # A2 = K_inv.dot(ones1)

        for ej in range(ei, dof):
            l, m = Vi[ej], Vj[ej]
            # ones2 = np.zeros_like(K)
            # ones2[l, m] = ones2[m, l] = 1.

            A = K_inv[[l, m]][:, [j, i]]
            B = K_inv[[i, j]][:, [m, l]]
            # B2 = K_inv.dot(ones2)

            # const = 1
            # if i == j:
            #     const *= 2
            # if l == m:
            #     const *= 2

            # assert np.allclose((A * B.T).sum(), const * (A2 * B2.T).sum())
            H[ei, ej] = H[ej, ei] = (A * B.T).sum()
            # H2[ei, ej] = H2[ej, ei] = (A2 * B2.T).sum()
    return - H * (delta - 2) / 2.


# if __name__ == '__main__':
#     K = make_sparse_spd_matrix(4)
#     D = make_spd_matrix(4)
#
#     assert np.allclose(first_derivative_h(D, K),
#                        first_derivative_h_version2(D, K))
