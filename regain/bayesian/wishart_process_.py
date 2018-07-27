import numpy as np
from scipy import linalg


def GWP_construct(umat, L, uut=None):
    """Build the sample from the GWP.

    Optimised with uut:
    uut = np.array([u.dot(u.T) for u in umat.T])
    """

    if uut is None:
        v, p, n = umat.shape
        M = np.zeros((p, p, n))
        for i in range(n):
            for j in range(v):
                Lu = L.dot(umat[j, :, i])
                LuuL = Lu[:, None].dot(Lu[None, :])
                M[..., i] += LuuL

    else:
        M = np.array([np.linalg.multi_dot((L, uu_i, L.T))
                      for uu_i in uut]).transpose()

    # assert np.allclose(N, M)
    return M


def predict(t_test, t_train, u_map, L_map, kern, inverse_width_map):
    """Predict covariance matrix for t_test.

    Parameters
    ----------
    t_{test, train} : ndarray
        Test and train time points.
    u_map : type
        MAP estimate of u parameter.
    L_map : type
        MAP estimate of L parameter.
    kern : type
        Kernel function.
    inverse_width_map : float
        MAP estimate of inverse_width, the kernel parameter.

    Returns
    -------
    ndarray, shape (p, p, t_test.size)
        List of covariances at t_test.

    """
    # Compute the mean for ustar for test data
    KB = kern(t_train[:, None], inverse_width=inverse_width_map)
    A = kern(t_test[:, None], t_train[:, None], inverse_width=inverse_width_map)
    invKB = linalg.pinvh(KB)

    # u_test is the mean of the data
    A_invKb = A.dot(invKB)

    u_test = np.tensordot(A_invKb, u_map.T, axes=1).T
    # equivalent to:
    # nu, p, _ = u_map.shape
    # u_test = np.zeros((nu, p, t_test.size))
    # for i in range(nu):
    #     for j in range(p):
    #         u_test[i, j, :] = A_invKb.dot(u_map[i, j, :])

    # Covariance of test data is
    # I_p - AK^{-1}A^T
    test_size = t_test.size
    test_covariance = np.eye(test_size) - A_invKb.dot(A.T)

    return GWP_construct(u_test, L_map)
