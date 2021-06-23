# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
from scipy import linalg
from sklearn.utils.extmath import fast_logdet


def h(D, K, delta=5):
    return -0.5 * (np.trace(K.T.dot(D)) - (delta - 2) * fast_logdet(K))


def first_derivative_h(D, K, delta=5):
    return -0.5 * (D - (delta - 2) * linalg.pinvh(K))


# def second_derivative_h(D, K, delta=5):
#     K_inv = linalg.pinvh(K)
#     return - 0.5 * (delta - 2) * K_inv.dot(K_inv)


def first_derivative_h_version2(D, K, delta=5):
    res = np.empty_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ones = np.zeros_like(D)
            ones[i, j] = ones[j, i] = 1.0
            res[i, j] = np.trace((D - (delta - 2) * linalg.pinvh(K)).dot(ones))
    return -res / 2.0


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
    return -H * (delta - 2) / 2.0
